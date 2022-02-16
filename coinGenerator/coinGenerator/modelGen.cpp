#include <cmath>
#include "modelGen.h"
#include "opencv2/imgproc.hpp"

//########## dwVec3D ##########

inline float dwVec3D::getLength() {
	return sqrt(x * x + y * y + z * z);
}

inline void dwVec3D::makeUnitVec() {
	float len = getLength();
	x /= len;
	y /= len;
	z /= len;
}

//########## modelGen ##########

void modelGen::setEdgeOnlyMode()
{
	m_bEdgeOnlyMode = true;
}

void modelGen::setImage(std::string& strFileName) {
	if (!m_originalImageGrayscale.empty())
		m_originalImageGrayscale.release();

	cv::Mat image = cv::imread(strFileName.c_str(), cv::IMREAD_GRAYSCALE);
	if (image.empty())
	{
		fprintf(stderr, "Fail to open the source image: ");
		fprintf(stderr, strFileName.c_str());
		fprintf(stderr, "\n");
		exit(ERROR_INVALID_IMAGE);
	}

	m_originalImageGrayscale = cv::Mat(image.rows + 1, image.cols + 1, CV_8UC1, cv::Scalar(0));
	if (m_originalImageGrayscale.empty())
	{
		fprintf(stderr, "Fail to create the source image.\n");
		exit(ERROR_INVALID_IMAGE);
	}

	//1�ȼ� Ȯ��
	uchar *pPixel, *pPixel2;
	for (int j = 0; j < image.rows; ++j)
	{
		pPixel = image.ptr<uchar>(j);
		pPixel2 = m_originalImageGrayscale.ptr<uchar>(j);
		for (int i = 0; i < image.cols; ++i)
		{
			pPixel2[i] = pPixel[i];
		}
	}

	for (int j = 0; j < image.rows; ++j)
	{
		pPixel = image.ptr<uchar>(j);
		pPixel2 = m_originalImageGrayscale.ptr<uchar>(j);

		int i = (image.cols - 1);
		pPixel2[i + 1] = pPixel[i];
	}

	{
		int j = (image.rows - 1);
		pPixel = image.ptr<uchar>(j);
		pPixel2 = m_originalImageGrayscale.ptr<uchar>(j + 1);

		for (int i = 0; i < image.cols; ++i)
		{
			*pPixel2++ = *pPixel++;
		}

		*pPixel2 = *(--pPixel);
	}

	m_etf.setImage(m_originalImageGrayscale);
	if(m_bEdgeOnlyMode)
		m_etf.smooth(6, 6);
	else
		m_etf.smooth(4, 4);
	m_etf.computeFDoG(1.f, 3.f, 0.99f);
	m_edgeImage = m_etf.getFDoGImage();

	cv::imwrite("res0.png", m_edgeImage);
	{
		cv::Mat tempImage = m_edgeImage.clone();

		uchar *pPixel1, *pPixel2;
		for (int j = 0; j < m_edgeImage.rows; ++j)
		{
			pPixel1 = m_edgeImage.ptr<uchar>(j);
			for (int i = 0; i < m_edgeImage.cols; ++i)
			{
				uchar val = *pPixel1++;
				if (val < 128)
				{
					for (int jj = -1; jj < 2; ++jj)
					{
						for (int ii = -1; ii < 2; ++ii)
						{
							if( (i+ii >=0) && (i + ii < m_nWidth) && (j + jj >= 0) && (j + jj < m_nHeight))
								tempImage.at<uchar>(j + jj, i + ii) = 0;
						}
					}
						
				}
			}
		}

		uchar* pPixel;
		for (int j = 0; j < tempImage.rows; ++j)
		{
			pPixel = tempImage.ptr<uchar>(j);
			for (int i = 0; i < tempImage.cols; ++i)
			{
				uchar val = *pPixel++;
				//�̺κ��� ������ �����϶��� �����ϵ��� �ٲٸ� ���� ����
				if (val < 244)
					m_seedsEdges.push_back(dwSeed(i, j));
			}
		}

		tempImage.release();
	}
	
	image.release();

	m_nWidth = m_originalImageGrayscale.cols;
	m_nHeight = m_originalImageGrayscale.rows;
	m_nResolution = m_nWidth * m_nHeight;
	m_nMaxDiagonal = sqrt(m_nWidth * m_nWidth + m_nHeight * m_nHeight) + 1;

//	setPixels();
//	setRelativeBrightness();
}

void modelGen::setImageRear(std::string& strFileName) {
	if (!m_rearImageGrayscale.empty())
		m_rearImageGrayscale.release();

	cv::Mat image = cv::imread(strFileName.c_str(), cv::IMREAD_GRAYSCALE);
	if (image.empty())
	{
		fprintf(stderr, "Fail to open the rear image: ");
		fprintf(stderr, strFileName.c_str());
		fprintf(stderr, "\n");
		exit(ERROR_INVALID_IMAGE);
	}

	m_rearImageGrayscale = cv::Mat(image.rows + 1, image.cols + 1, CV_8UC1, cv::Scalar(0));
	if (m_rearImageGrayscale.empty())
	{
		fprintf(stderr, "Fail to create the source image.\n");
		exit(ERROR_INVALID_IMAGE);
	}

	//1�ȼ� Ȯ��
	uchar* pPixel, * pPixel2;
	for (int j = 0; j < image.rows; ++j)
	{
		pPixel = image.ptr<uchar>(j);
		pPixel2 = m_rearImageGrayscale.ptr<uchar>(j);
		for (int i = 0; i < image.cols; ++i)
		{
			pPixel2[i] = pPixel[i];
		}
	}

	for (int j = 0; j < image.rows; ++j)
	{
		pPixel = image.ptr<uchar>(j);
		pPixel2 = m_rearImageGrayscale.ptr<uchar>(j);

		int i = (image.cols - 1);
		pPixel2[i + 1] = pPixel[i];
	}

	{
		int j = (image.rows - 1);
		pPixel = image.ptr<uchar>(j);
		pPixel2 = m_rearImageGrayscale.ptr<uchar>(j + 1);

		for (int i = 0; i < image.cols; ++i)
		{
			*pPixel2++ = *pPixel++;
		}

		*pPixel2 = *(--pPixel);
	}


	image.release();
}

void modelGen::setMask(std::string& strFileName)
{
	if (m_bEdgeOnlyMode)
	{
		return;
	}

	cv::Mat image = cv::imread(strFileName.c_str(), cv::IMREAD_GRAYSCALE);
	if (image.empty())
	{
		fprintf(stderr, "Fail to open the mask image: ");
		fprintf(stderr, strFileName.c_str());
		fprintf(stderr, "\n");
		exit(ERROR_INVALID_IMAGE);
	}

	cv::Mat maskImage = cv::Mat(image.rows + 1, image.cols + 1, CV_8UC1, cv::Scalar(0));
	if (maskImage.empty())
	{
		fprintf(stderr, "Fail to create the mask image.\n");
		exit(ERROR_INVALID_IMAGE);
	}

	//1�ȼ� Ȯ��
	uchar* pPixel, * pPixel2;
	for (int j = 0; j < image.rows; ++j)
	{
		pPixel = image.ptr<uchar>(j);
		pPixel2 = maskImage.ptr<uchar>(j);
		for (int i = 0; i < image.cols; ++i)
		{
			pPixel2[i] = pPixel[i];
		}
	}

	for (int j = 0; j < image.rows; ++j)
	{
		pPixel = image.ptr<uchar>(j);
		pPixel2 = maskImage.ptr<uchar>(j);

		int i = (image.cols - 1);
		pPixel2[i + 1] = pPixel[i];
	}

	{
		int j = (image.rows - 1);
		pPixel = image.ptr<uchar>(j);
		pPixel2 = maskImage.ptr<uchar>(j + 1);

		for (int i = 0; i < image.cols; ++i)
		{
			*pPixel2++ = *pPixel++;
		}

		*pPixel2 = *(--pPixel);
	}
	image.release();

	//�ʿ� ����ũ �̹��� ����
	m_pFacialComponentMap = new unsigned char[m_nResolution];
	pPixel = maskImage.data;
	unsigned char* pMapPixel = m_pFacialComponentMap;
	for (int i = 0; i < m_nResolution; ++i)
	{
		*pMapPixel++ = *pPixel++;
	}

	//�� ���� Ȯ��
	expandEyeRegionVer2();
}

void modelGen::setNoseDorsum(std::string& strFileName)
{
	if (m_bEdgeOnlyMode)
	{
		return;
	}

	const int nBeginning = 3 * 27;
	const int nMemSize = 3 * 4;
	float* pMem = new float[nMemSize];
	FILE * filePointer;
	filePointer = fopen(strFileName.c_str(), "rb");
	if (filePointer == NULL)
	{
		fprintf(stderr, "Fail to open the nose dorsum file: ");
		fprintf(stderr, strFileName.c_str());
		fprintf(stderr, "\n");
		exit(ERROR_FILE_OPEN);
	}
	fseek(filePointer, nBeginning * sizeof(float), SEEK_SET);
	fread(pMem, sizeof(float), nMemSize, filePointer);
	fclose(filePointer);

	int index = 0;
	m_noseDorsumFeatures[0] = dwVec3D(pMem[index], pMem[index + 1], pMem[index + 2]);
	index += 3;
	m_noseDorsumFeatures[1] = dwVec3D(pMem[index], pMem[index + 1], pMem[index + 2]);
	index += 3;
	m_noseDorsumFeatures[2] = dwVec3D(pMem[index], pMem[index + 1], pMem[index + 2]);
	index += 3;
	m_noseDorsumFeatures[3] = dwVec3D(pMem[index], pMem[index + 1], pMem[index + 2]);

	delete[] pMem;
}

void modelGen::setTextImage(std::string& strFileName)
{
	cv::Mat image = cv::imread(strFileName.c_str(), cv::IMREAD_GRAYSCALE);
	if (image.empty())
	{
		fprintf(stderr, "Fail to open the text image: ");
		fprintf(stderr, strFileName.c_str());
		fprintf(stderr, "\n");
		exit(ERROR_INVALID_IMAGE);
	}

	m_pTextMap = new bool[m_nResolution];
	for (int i = 0; i < m_nResolution; ++i)
	{
		m_pTextMap[i] = false;
	}

	uchar* pPixel = image.data;
	int index = 0;
	for (int j = 0; j < image.rows; ++j)
	{
		for (int i = 0; i < image.cols; ++i)
		{
			unsigned char intensity = pPixel[index++];
			if (intensity < 128)
				m_pTextMap[j * m_nWidth + i] = true;
		}
	}

	/*
	cv::Mat image2(m_nWidth, m_nHeight, CV_8UC1, cv::Scalar(0));
	uchar* pPixel2 = image2.data;
	for (int index = 0; index < m_nResolution; ++index)
	{
		if(m_pTextMap[index])
			pPixel2[index] = 0;
		else
			pPixel2[index] = 255;
	}
	cv::imwrite("res_text.png", image2);
	*/
}

void modelGen::expandEyeRegionVer1(int nRadius)
{
	int nRadiusSquare = nRadius * nRadius;
	unsigned char* tempFacialComponentMap = new unsigned char[m_nResolution];

	int index = 0;
	for (int j = 0; j < m_nHeight; ++j)
	{
		for (int i = 0; i < m_nWidth; ++i)
		{
			tempFacialComponentMap[index] = m_pFacialComponentMap[index];
			if ((m_pFacialComponentMap[index] < facialComponent::neck) && (m_pFacialComponentMap[index] != facialComponent::eyeL) && (m_pFacialComponentMap[index] != facialComponent::eyeR))
			{
				bool bNotFound = true;
				for (int jj = -nRadius; bNotFound && (jj <= nRadius); ++jj)
				{
					for (int ii = -nRadius; bNotFound && (ii <= nRadius); ++ii)
					{
						if (((i + ii) >= 0) && ((i + ii) < m_nWidth) && ((j + jj) >= 0) && ((j + jj) < m_nHeight))
						{
							int index2 = m_nWidth * (j + jj) + (i + ii);
							if ((m_pFacialComponentMap[index2] == facialComponent::eyeL || m_pFacialComponentMap[index2] == facialComponent::eyeR) && ((ii * ii + jj * jj) <= nRadiusSquare))
							{
								tempFacialComponentMap[index] = m_pFacialComponentMap[index2];
								bNotFound = false;
							}
						}
					}
				}
			}
			++index;
		}
	}

	unsigned char* pMapPixel = m_pFacialComponentMap;
	unsigned char* pMapPixel2 = tempFacialComponentMap;
	for (int i = 0; i < m_nResolution; ++i)
	{
		*pMapPixel++ = *pMapPixel2++;
	}
	delete[] tempFacialComponentMap;
}

void modelGen::expandEyeRegionVer2()
{
	const int &nRadius = m_nEyeExpansion;
	int nRadiusSquare = nRadius * nRadius;
	unsigned char* tempFacialComponentMap = new unsigned char[m_nResolution];

	int index = 0;
	for (int j = 0; j < m_nHeight; ++j)
	{
		for (int i = 0; i < m_nWidth; ++i)
		{
			tempFacialComponentMap[index] = m_pFacialComponentMap[index];
			if ((m_pFacialComponentMap[index] == facialComponent::eyeL) || (m_pFacialComponentMap[index] == facialComponent::eyeR))
			{
				for (int jj = -nRadius; jj <= nRadius; ++jj)
				{
					for (int ii = -nRadius; ii <= nRadius; ++ii)
					{
						if (((i + ii) >= 0) && ((i + ii) < m_nWidth) && ((j + jj) >= 0) && ((j + jj) < m_nHeight))
						{
							int index2 = m_nWidth * (j + jj) + (i + ii);
							if ((tempFacialComponentMap[index2] != facialComponent::eyeL) && (tempFacialComponentMap[index2] != facialComponent::eyeR) && ((ii * ii + jj * jj) <= nRadiusSquare))
							{
								tempFacialComponentMap[index2] = m_pFacialComponentMap[index];
							}
						}
					}
				}
			}
			++index;
		}
	}

	unsigned char* pMapPixel = m_pFacialComponentMap;
	unsigned char* pMapPixel2 = tempFacialComponentMap;
	for (int i = 0; i < m_nResolution; ++i)
	{
		*pMapPixel++ = *pMapPixel2++;
	}
	delete[] tempFacialComponentMap;
}

void modelGen::preComputeHeightMap()
{
	if (m_bEdgeOnlyMode)
	{
		initDistanceFilter();
		computeDistanceMapEdgeOnly();
	}
	else
	{
		initDistanceFilter();
		generateFacialComponentBoundary();
		generateNoseDorsumLine();
		computeDistanceMap();
	}
	
}

void modelGen::generateNoseDorsumLine()
{
	for (int i = 0; i < 3; ++i)
	{
		int curX, curY;

		int x1 = m_noseDorsumFeatures[i].x;
		int y1 = m_noseDorsumFeatures[i].y;
		int x2 = m_noseDorsumFeatures[i + 1].x;
		int y2 = m_noseDorsumFeatures[i + 1].y;

		int dx = x2 - x1;
		int dy = y2 - y1;
		int steps;
		float xIncrement, yIncrement, x = x1, y = y1;

		if (fabs(dx) > fabs(dy))
			steps = fabs(dx);
		else
			steps = fabs(dy);

		xIncrement = float(dx) / float(steps);
		yIncrement = float(dy) / float(steps);
		curX = round(x);
		curY = round(y);
		m_seedsNoseDorsum.push_back(dwSeed(curX, curY));
		for (int k = 0; k < steps; k++) 
		{
			x += xIncrement;
			y += yIncrement;
			curX = round(x);
			curY = round(y);
			m_seedsNoseDorsum.push_back(dwSeed(curX, curY));
		}
	}
	/*
	cv::Mat image(m_nWidth, m_nHeight, CV_8UC1, cv::Scalar(0));
	uchar* pPixel = image.data;
	for (int index = 0; index < m_nResolution; ++index)
	{
		pPixel[index] = 255;
	}
	for (int i = 0; i < m_seedsNoseDorsum.size(); ++i)
	{
		dwSeed seed = m_seedsNoseDorsum[i];
		pPixel[seed.getY() * m_nWidth + seed.getX()] = 0;
	}
	cv::imwrite("res_nose.png", image);
	*/
}

void modelGen::initDistanceFilter()
{
	
	m_pDistanceFilter = new float[m_nFilterWidth * m_nFilterWidth];

	auto getDist = [](int x, int y) {
		return float(x * x + y * y);
	};

	for (int j = 0; j < m_nFilterWidth; ++j)
	{
		for (int i = 0; i < m_nFilterWidth; ++i)
		{
			m_pDistanceFilter[j * m_nFilterWidth + i] = getDist(i - m_nFilterCentre, j - m_nFilterCentre);
		}
	}
}

void modelGen::generateFacialComponentBoundary()
{
	auto isHeadRegion = [](unsigned char comp) {
		return (((comp > facialComponent::background) && (comp < facialComponent::neck)) || (comp == facialComponent::hat) || (comp == facialComponent::hair));
	};

	auto isEyebrowsLipsHairRegion = [](unsigned char comp) {
		return ((comp == facialComponent::eyebrowL) || (comp == facialComponent::eyebrowR) || (comp == facialComponent::lipU) || (comp == facialComponent::lipD) || (comp == facialComponent::hair) || (comp == facialComponent::hat));
	};

	auto isFacialRegion = [](unsigned char comp) {
		return ((comp > facialComponent::background) && (comp < facialComponent::neck));
	};

	auto isNoseRegion = [](unsigned char comp) {
		return (comp == facialComponent::nose);
	};

	m_pFacialComponentBoundaryMap = new unsigned char[m_nResolution];

	int index = 0;
	for (int j = 0; j < m_nHeight; ++j)
	{
		for (int i = 0; i < m_nWidth; ++i)
		{
			m_pFacialComponentBoundaryMap[index] = 0;
			unsigned char currentComponent = m_pFacialComponentMap[index];

			bool bHeadNotFound = true;
			bool bHEyebrowsNotFound = true;
			bool bNoseNotFound = true;

			for (int jj = -1; (bHeadNotFound || bHEyebrowsNotFound || bNoseNotFound) && (jj <= 1); ++jj)
			{
				for (int ii = -1; (bHeadNotFound || bHEyebrowsNotFound || bNoseNotFound) && (ii <= 1); ++ii)
				{
					if ((ii || jj) && ((i + ii) >= 0) && ((i + ii) < m_nWidth) && ((j + jj) >= 0) && ((j + jj) < m_nHeight))
					{
						int index2 = m_nWidth * (j + jj) + (i + ii);
						unsigned char neighborComponent = m_pFacialComponentMap[index2];
						
						if (bHeadNotFound && (isHeadRegion(currentComponent) != isHeadRegion(neighborComponent)))
						{
							m_pFacialComponentBoundaryMap[index] |= unsigned char(0b10000000);	//head������ �ٿ������ MSB�� 1�� ����
							m_seedsBoundaryHead.push_back(dwSeed(i,j));
							bHeadNotFound = false;
						}

						if (bHEyebrowsNotFound && (currentComponent != neighborComponent) && (isEyebrowsLipsHairRegion(neighborComponent)))
						{
							m_pFacialComponentBoundaryMap[index] |= unsigned char(0b01000000);	//����, �Լ�, ��� ������ �ٿ������ �ι�° ��Ʈ�� 1�� ����
							m_seedsBoundaryEyebrow.push_back(dwSeed(i, j));
							bHEyebrowsNotFound = false;
						}

						if (bNoseNotFound && isFacialRegion(currentComponent) && !isNoseRegion(currentComponent) && isNoseRegion(neighborComponent))
						{
							m_pFacialComponentBoundaryMap[index] |= unsigned char(0b00100000);	//�� ������ �ٿ������ ����° ��Ʈ�� 1�� ����
							m_seedsBoundaryNose.push_back(dwSeed(i, j));
							bNoseNotFound = false;
						}
					}
				}
			}
			++index;
		}
	}
}

void modelGen::computeDistanceMap()
{
	m_pDistanceMapEdge = new float[m_nResolution];
	m_pDistanceMapHead = new float[m_nResolution];
	m_pDistanceMapEyebrow = new float[m_nResolution];
	m_pDistanceMapNose = new float[m_nResolution];
	m_pDistanceMapNoseDorsum = new float[m_nResolution];
	m_pIndexMapNoseDorsum = new int[m_nResolution];

	for (int i = 0; i < m_nResolution; ++i)
	{
		m_pDistanceMapEdge[i] = m_pDistanceMapHead[i] = m_pDistanceMapEyebrow[i] = m_pDistanceMapNose[i] = m_pDistanceMapNoseDorsum[i] = FLT_MAX;
		m_pIndexMapNoseDorsum[i] = -1;
	}

	int nSize = m_seedsEdges.size();
	for (int k = 0; k < nSize; ++k)
	{
		const dwSeed& seed = m_seedsEdges[k];
		int i = 0;
		int j = 0;
		for (int index = 0; index < m_nResolution; ++index)
		{
			int diffX = i - seed.getX();
			int diffY = j - seed.getY();
			int indexFilter = (m_nFilterCentre + diffY) * m_nFilterWidth + (m_nFilterCentre + diffX);
			if (m_pDistanceMapEdge[index] > m_pDistanceFilter[indexFilter])
			{
				m_pDistanceMapEdge[index] = m_pDistanceFilter[indexFilter];
			}

			if ((++i) == m_nWidth)
			{
				i = 0;
				++j;
			}
		}
	}

	nSize = m_seedsBoundaryHead.size();
	for (int k = 0; k < nSize; ++k)
	{
		const dwSeed& seed = m_seedsBoundaryHead[k];
		int i = 0;
		int j = 0;
		for (int index = 0; index < m_nResolution; ++index)
		{
			int diffX = i - seed.getX();
			int diffY = j - seed.getY();
			int indexFilter = (m_nFilterCentre + diffY) * m_nFilterWidth + (m_nFilterCentre + diffX);
			if (m_pDistanceMapHead[index] > m_pDistanceFilter[indexFilter])
			{
				m_pDistanceMapHead[index] = m_pDistanceFilter[indexFilter];
			}

			if ((++i) == m_nWidth)
			{
				i = 0;
				++j;
			}
		}
	}

	nSize = m_seedsBoundaryEyebrow.size();
	for (int k = 0; k < nSize; ++k)
	{
		const dwSeed& seed = m_seedsBoundaryEyebrow[k];
		int i = 0;
		int j = 0;
		for (int index = 0; index < m_nResolution; ++index)
		{
			int diffX = i - seed.getX();
			int diffY = j - seed.getY();
			int indexFilter = (m_nFilterCentre + diffY) * m_nFilterWidth + (m_nFilterCentre + diffX);
			if (m_pDistanceMapEyebrow[index] > m_pDistanceFilter[indexFilter])
			{
				m_pDistanceMapEyebrow[index] = m_pDistanceFilter[indexFilter];
			}

			if ((++i) == m_nWidth)
			{
				i = 0;
				++j;
			}
		}
	}

	nSize = m_seedsNoseDorsum.size();
	for (int k = 0; k < nSize; ++k)
	{
		const dwSeed& seed = m_seedsNoseDorsum[k];
		int i = 0;
		int j = 0;
		for (int index = 0; index < m_nResolution; ++index)
		{
			int diffX = i - seed.getX();
			int diffY = j - seed.getY();
			int indexFilter = (m_nFilterCentre + diffY) * m_nFilterWidth + (m_nFilterCentre + diffX);
			if (m_pDistanceMapNoseDorsum[index] > m_pDistanceFilter[indexFilter])
			{
				m_pDistanceMapNoseDorsum[index] = m_pDistanceFilter[indexFilter];
				m_pIndexMapNoseDorsum[index] = k;
			}

			if ((++i) == m_nWidth)
			{
				i = 0;
				++j;
			}
		}
	}

	int * pNoseBoundaryIndexMap = new int[m_nResolution];

	nSize = m_seedsBoundaryNose.size();
	for (int k = 0; k < nSize; ++k)
	{
		const dwSeed& seed = m_seedsBoundaryNose[k];
		int i = 0;
		int j = 0;

		int nNoseDorsumIndex = seed.getY() * m_nWidth + seed.getX();
		float fDistCorrection = -m_pDistanceMapNoseDorsum[nNoseDorsumIndex];

		for (int index = 0; index < m_nResolution; ++index)
		{
			int diffX = i - seed.getX();
			int diffY = j - seed.getY();
			int indexFilter = (m_nFilterCentre + diffY) * m_nFilterWidth + (m_nFilterCentre + diffX);

			
			if (m_pDistanceMapNose[index] > (m_pDistanceFilter[indexFilter] + fDistCorrection))
			{
				m_pDistanceMapNose[index] = (m_pDistanceFilter[indexFilter] + fDistCorrection);
				pNoseBoundaryIndexMap[index] = k;
			}

			if ((++i) == m_nWidth)
			{
				i = 0;
				++j;
			}
		}
	}

	int i = 0;
	int j = 0;
	for (int index = 0; index < m_nResolution; ++index)
	{
		int nSeedIndex = pNoseBoundaryIndexMap[index];
		const dwSeed& seed = m_seedsBoundaryNose[nSeedIndex];

		int diffX = i - seed.getX();
		int diffY = j - seed.getY();
		int indexFilter = (m_nFilterCentre + diffY) * m_nFilterWidth + (m_nFilterCentre + diffX);

		m_pDistanceMapNose[index] = m_pDistanceFilter[indexFilter];

		if ((++i) == m_nWidth)
		{
			i = 0;
			++j;
		}
	}

	delete[] pNoseBoundaryIndexMap;
	
	cv::Mat image1(m_nWidth, m_nHeight, CV_8UC1, cv::Scalar(0));
	cv::Mat image2(m_nWidth, m_nHeight, CV_8UC1, cv::Scalar(0));
	cv::Mat image3(m_nWidth, m_nHeight, CV_8UC1, cv::Scalar(0));
	cv::Mat image4(m_nWidth, m_nHeight, CV_8UC1, cv::Scalar(0));
	uchar* pPixel1 = image1.data;
	uchar* pPixel2 = image2.data;
	uchar* pPixel3 = image3.data;
	uchar* pPixel4 = image4.data;
	for (int index = 0; index < m_nResolution; ++index)
	{
		pPixel1[index] = uchar(m_pDistanceMapHead[index] > 255 ? 255 : m_pDistanceMapHead[index]);
		pPixel2[index] = uchar(m_pDistanceMapEyebrow[index] > 255 ? 255 : m_pDistanceMapEyebrow[index]);
		pPixel3[index] = uchar(m_pDistanceMapNose[index] > 255 ? 255 : m_pDistanceMapNose[index]);
		pPixel4[index] = uchar(m_pDistanceMapNoseDorsum[index] > 255 ? 255 : m_pDistanceMapNoseDorsum[index]);
	}
	nSize = m_seedsNoseDorsum.size();
	for (int k = 0; k < nSize; ++k)
	{
		const dwSeed& seed = m_seedsNoseDorsum[k];
		int i = seed.getX();
		int j = seed.getY();

		int index = j * m_nWidth + i;
		pPixel3[index] = uchar(128);
		pPixel4[index] = uchar(128);
	}
	
	nSize = m_seedsBoundaryNose.size();
	for (int k = 0; k < nSize; ++k)
	{
		const dwSeed& seed = m_seedsBoundaryNose[k];
		int i = seed.getX();
		int j = seed.getY();

		int index = j * m_nWidth + i;
		pPixel3[index] = uchar(128);
		pPixel4[index] = uchar(128);
	}
	
	cv::imwrite("res1.png", image1);
	cv::imwrite("res2.png", image2);
	cv::imwrite("res3.png", image3);
	cv::imwrite("res4.png", image4);
	
}

void modelGen::computeDistanceMapEdgeOnly()
{
	m_pDistanceMapEdge = new float[m_nResolution];
	
	for (int i = 0; i < m_nResolution; ++i)
	{
		m_pDistanceMapEdge[i] = FLT_MAX;
	}

	int nSize = m_seedsEdges.size();
	for (int k = 0; k < nSize; ++k)
	{
		const dwSeed& seed = m_seedsEdges[k];
		int i = 0;
		int j = 0;
		for (int index = 0; index < m_nResolution; ++index)
		{
			int diffX = i - seed.getX();
			int diffY = j - seed.getY();
			int indexFilter = (m_nFilterCentre + diffY) * m_nFilterWidth + (m_nFilterCentre + diffX);
			if (m_pDistanceMapEdge[index] > m_pDistanceFilter[indexFilter])
			{
				m_pDistanceMapEdge[index] = m_pDistanceFilter[indexFilter];
			}

			if ((++i) == m_nWidth)
			{
				i = 0;
				++j;
			}
		}
	}
}

void modelGen::computeIfIsInsideRadius()
{
	int nRadius = m_nWidth >> 1;
	int nSquaredRadius = nRadius * nRadius;
	m_pIsInsideRadiusMap = new bool[m_nResolution];

	for (int j = 0; j < nRadius; ++j)
	{
		for (int i = 0; i < nRadius; ++i)
		{
			int index1 = j * m_nWidth + i;
			int index2 = j * m_nWidth + (m_nWidth - 1 - i);
			int index3 = (m_nHeight - 1 - j) * m_nWidth + i;
			int index4 = (m_nHeight - 1 - j) * m_nWidth + (m_nWidth - 1 - i);
			int nSquareDistFromCenter = (i - nRadius) * (i - nRadius) + (j - nRadius) * (j - nRadius);
			if (nSquareDistFromCenter <= nSquaredRadius)
			{
				m_pIsInsideRadiusMap[index1] = true;
				m_pIsInsideRadiusMap[index2] = true;
				m_pIsInsideRadiusMap[index3] = true;
				m_pIsInsideRadiusMap[index4] = true;
			}
			else
			{
				m_pIsInsideRadiusMap[index1] = false;
				m_pIsInsideRadiusMap[index2] = false;
				m_pIsInsideRadiusMap[index3] = false;
				m_pIsInsideRadiusMap[index4] = false;
			}
		}
	}
}

void modelGen::generate3Dmodel(std::string& strFileName, float fProductRadius)
{
	// ���� �������� üũ
	computeIfIsInsideRadius();

	//����Ʈ�� ���
	if(m_bEdgeOnlyMode)
		computeHeightMapEdgeOnly();
	else
		computeHeightMap();

	// LOD ����
	computeLOD();

	//�� �ٿ���� ����
	generateBoundaryCircle();

	//Delaunay Triangulation ����
	tessellate();

	//����Ʈ�� ���(rear)
	computeHeightMapRear();

	// LOD ����(rear)
	computeLODRear();

	//�� �ٿ���� ����(rear)
	generateBoundaryCircleRear();

	//Delaunay Triangulation ����(rear)
	tessellateRear();

	//stl ���Ϸ� export
	exportModel(strFileName, fProductRadius);
}

void modelGen::exportModel(std::string& strFileName, float fProductRadius)
{
	float fScale = fProductRadius / (m_nHeight >> 1);

	int nNumSideFaces = m_nBoundaryVerticesNum * 2;

	int nHeaderSize = 80 + 4;
	unsigned int nFaceNum = getFacesNum();
	unsigned int nFaceNumRear = getFacesNumRear();
	//unsigned int nFaceNumRear = 0;
	int nFileSize = nHeaderSize + (12 * 4 + 2) * (nFaceNum + nFaceNumRear + nNumSideFaces);
	
	int indexMem = 80;
	char* pMem = new char[nFileSize];
	
	*(unsigned int*)(pMem + indexMem) = (nFaceNum + nFaceNumRear + nNumSideFaces);
	indexMem += 4;

	int vtxIndex[3];
	for (int index = 0; index < nFaceNum; ++index)
	{
		vtxIndex[0] = m_pFaces->tris[index * 3 + 0];
		vtxIndex[1] = m_pFaces->tris[index * 3 + 1];
		vtxIndex[2] = m_pFaces->tris[index * 3 + 2];

		float x[3];
		float y[3];
		float z[3];

		for (int i = 0; i < 3; ++i)
		{
			x[i] = m_pFaces->points[vtxIndex[i]].x;
			y[i] = m_pFaces->points[vtxIndex[i]].y;
			if (vtxIndex[i] >= (getVerticesNum() - m_nBoundaryVerticesNum))
			{
				z[i] = 0.f;
			}
			else
			{
				int idx = int(x[i] + (float(m_nWidth) * 0.5f)) + int(y[i] + (float(m_nWidth) * 0.5f)) * m_nWidth;
				z[i] = m_pHeightMap[idx];
			}

		}

		dwVec3D vec1(x[1] - x[0], y[1] - y[0], z[1] - z[0]);
		dwVec3D vec2(x[2] - x[1], y[2] - y[1], z[2] - z[1]);
		vec1.makeUnitVec();
		vec2.makeUnitVec();

		dwVec3D normal = vec1.crossProduct(vec2);
		normal.makeUnitVec();

		//triangle1
		*(float*)(pMem + indexMem) = normal.x;
		indexMem += 4;
		*(float*)(pMem + indexMem) = normal.y;
		indexMem += 4;
		*(float*)(pMem + indexMem) = normal.z;
		indexMem += 4;

		for (int i = 0; i < 3; ++i)
		{
			*(float*)(pMem + indexMem) = x[i] * fProductRadius;
			indexMem += 4;
			*(float*)(pMem + indexMem) = y[i] * fProductRadius;
			indexMem += 4;
			*(float*)(pMem + indexMem) = z[i] * fProductRadius;
			indexMem += 4;
		}

		*(unsigned short*)(pMem + indexMem) = 0;
		indexMem += 2;
	}

	for (int index = 0; index < nFaceNumRear; ++index)
	{
		vtxIndex[0] = m_pFacesRear->tris[index * 3 + 0];
		vtxIndex[1] = m_pFacesRear->tris[index * 3 + 1];
		vtxIndex[2] = m_pFacesRear->tris[index * 3 + 2];

		float x[3];
		float y[3];
		float z[3];

		for (int i = 0; i < 3; ++i)
		{
			x[i] = m_pFacesRear->points[vtxIndex[i]].x;
			y[i] = m_pFacesRear->points[vtxIndex[i]].y;
			if (vtxIndex[i] >= (getVerticesNumRear() - m_nBoundaryVerticesNum))
			{
				z[i] = 0.f;
			}
			else
			{
				int idx = int(x[i] + (float(m_nWidth) * 0.5f)) + int(y[i] + (float(m_nWidth) * 0.5f)) * m_nWidth;
				z[i] = m_pHeightMapRear[idx];
			}

			x[i] = -x[i];
			z[i] = -z[i] - 20.f;

		}

		dwVec3D vec1(x[1] - x[0], y[1] - y[0], z[1] - z[0]);
		dwVec3D vec2(x[2] - x[1], y[2] - y[1], z[2] - z[1]);
		vec1.makeUnitVec();
		vec2.makeUnitVec();

		dwVec3D normal = vec1.crossProduct(vec2);
		normal.makeUnitVec();

		//triangle1
		*(float*)(pMem + indexMem) = normal.x;
		indexMem += 4;
		*(float*)(pMem + indexMem) = normal.y;
		indexMem += 4;
		*(float*)(pMem + indexMem) = normal.z;
		indexMem += 4;

		for (int i = 0; i < 3; ++i)
		{
			*(float*)(pMem + indexMem) = x[i] * fProductRadius;
			indexMem += 4;
			*(float*)(pMem + indexMem) = y[i] * fProductRadius;
			indexMem += 4;
			*(float*)(pMem + indexMem) = z[i] * fProductRadius;
			indexMem += 4;
		}

		*(unsigned short*)(pMem + indexMem) = 0;
		indexMem += 2;
	}

	int nExpandedRaius = (m_nWidth >> 1) + 1;
	float fDegree = 2.5f;						//5�� ���� (degree)
	float fAngle = 3.141592f / 180 * fDegree;	//5�� ���� (radian)

	float x = 0.f;
	float y = nExpandedRaius;

	float prevX = x;
	float prevY = y;
	for (int index = 0; index < nNumSideFaces; ++index)
	{
		float nextX = x * cos(((index >> 1) + 1) * fAngle) - y * sin(((index >> 1) + 1) * fAngle);
		float nextY = x * sin(((index >> 1) + 1) * fAngle) + y * cos(((index >> 1) + 1) * fAngle);
		
		std::vector<dwVec3D> vtxList;
		if (index % 2 == 0)
		{
			vtxList.push_back(dwVec3D(prevX, prevY, 0));
			vtxList.push_back(dwVec3D(prevX, prevY, -20));
			vtxList.push_back(dwVec3D(nextX, nextY, -20));
		}
		else
		{
			vtxList.push_back(dwVec3D(nextX, nextY, -20));
			vtxList.push_back(dwVec3D(nextX, nextY, 0));
			vtxList.push_back(dwVec3D(prevX, prevY, 0));
			prevX = nextX;
			prevY = nextY;
		}
		
		float x[3];
		float y[3];
		float z[3];

		for (int i = 0; i < 3; ++i)
		{
			x[i] = vtxList[i].x;
			y[i] = vtxList[i].y;
			z[i] = vtxList[i].z;
		}

		dwVec3D vec1(x[1] - x[0], y[1] - y[0], z[1] - z[0]);
		dwVec3D vec2(x[2] - x[1], y[2] - y[1], z[2] - z[1]);
		vec1.makeUnitVec();
		vec2.makeUnitVec();

		dwVec3D normal = vec1.crossProduct(vec2);
		normal.makeUnitVec();

		//triangle1
		*(float*)(pMem + indexMem) = normal.x;
		indexMem += 4;
		*(float*)(pMem + indexMem) = normal.y;
		indexMem += 4;
		*(float*)(pMem + indexMem) = normal.z;
		indexMem += 4;

		for (int i = 0; i < 3; ++i)
		{
			*(float*)(pMem + indexMem) = x[i] * fProductRadius;
			indexMem += 4;
			*(float*)(pMem + indexMem) = y[i] * fProductRadius;
			indexMem += 4;
			*(float*)(pMem + indexMem) = z[i] * fProductRadius;
			indexMem += 4;
		}

		*(unsigned short*)(pMem + indexMem) = 0;
		indexMem += 2;
	}

	FILE* pFile;
	int err = fopen_s(&pFile, strFileName.c_str(), "wb");
	if (err != 0)
	{
		fprintf(stderr, "Fail to create the model file: ");
		fprintf(stderr, strFileName.c_str());
		fprintf(stderr, "\n");
		exit(ERROR_FILE_OPEN);
	}

	fwrite(pMem, sizeof(char), nFileSize, pFile);
	fclose(pFile);
}

void modelGen::computeLOD()
{
	bool* pSuccessMap = new bool[m_nResolution];
	m_pRemovable = new bool[m_nResolution];

	for (int i = 0; i < m_nResolution; ++i)
	{
		pSuccessMap[i] = true;
		m_pRemovable[i] = false;
	}

	auto isFourPartsFlat = [=](const std::vector<int>& xpos, const std::vector<int>& ypos) {
		return (pSuccessMap[xpos[0] + m_nWidth * ypos[0]] && pSuccessMap[xpos[1] + m_nWidth * ypos[1]] &&
			pSuccessMap[xpos[3] + m_nWidth * ypos[3]] && pSuccessMap[xpos[4] + m_nWidth * ypos[4]]);
	};

	auto isFourPartsOutside = [=](const std::vector<int>& xpos, const std::vector<int>& ypos) {
		return (!m_pIsInsideRadiusMap[xpos[0] + m_nWidth * ypos[0]] && !m_pIsInsideRadiusMap[xpos[1] + m_nWidth * ypos[1]] &&
			!m_pIsInsideRadiusMap[xpos[3] + m_nWidth * ypos[3]] && !m_pIsInsideRadiusMap[xpos[4] + m_nWidth * ypos[4]]);
	};

	for (int level = 2; level < 512; level *= 2)
	{
		int halflevel = level >> 1;

		for (int jbegin = 0; jbegin < (m_nHeight - 1); jbegin += level)
		{
			for (int ibegin = 0; ibegin < (m_nWidth - 1); ibegin += level)
			{
				std::vector<int> xpos;	//9���� ���ؽ� x��ǥ
				std::vector<int> ypos;	//9���� ���ؽ� y��ǥ
				for (int j = jbegin; j <= (jbegin + level); j += halflevel)
				{
					for (int i = ibegin; i <= (ibegin + level); i += halflevel)
					{
						xpos.push_back(i);
						ypos.push_back(j);
					}
				}

				//���� 4���� ������ ���ؽ��� ��� radius�ٱ��̶�� 
				if (isFourPartsOutside(xpos, ypos))
				{

				}
				else
				{
					if (isFourPartsFlat(xpos, ypos))	//4���� ������Ʈ�� ���� �������� faltten �Ǿ��ٸ� ture, ���� �߰��� flatten ����
					{
						bool bAllRemovable = true;

						//��ܰ� ���� ������ ���� �˻�� ���� �������� �ߺ��ǹǷ� �̹� �˻��ߴٸ� pass
						if (m_pRemovable[m_nWidth * ypos[1] + xpos[1]] || isVertexRemovable(xpos[0], ypos[0], xpos[2], ypos[2], xpos[1], ypos[1]))
							m_pRemovable[m_nWidth * ypos[1] + xpos[1]] = true;
						else
							bAllRemovable = false;

						//��ܰ� ���� ������ ���� �˻�� ���� �������� �ߺ��ǹǷ� �̹� �˻��ߴٸ� pass
						if (m_pRemovable[m_nWidth * ypos[3] + xpos[3]] || isVertexRemovable(xpos[0], ypos[0], xpos[6], ypos[6], xpos[3], ypos[3]))
							m_pRemovable[m_nWidth * ypos[3] + xpos[3]] = true;
						else
							bAllRemovable = false;

						if (isVertexRemovable(xpos[2], ypos[2], xpos[8], ypos[8], xpos[5], ypos[5]))
							m_pRemovable[m_nWidth * ypos[5] + xpos[5]] = true;
						else
							bAllRemovable = false;

						if (isVertexRemovable(xpos[6], ypos[6], xpos[8], ypos[8], xpos[7], ypos[7]))
							m_pRemovable[m_nWidth * ypos[7] + xpos[7]] = true;
						else
							bAllRemovable = false;

						if (isVertexRemovable(xpos[0], ypos[0], xpos[2], ypos[2], xpos[6], ypos[6], xpos[8], ypos[8], xpos[4], ypos[4]))
							m_pRemovable[m_nWidth * ypos[4] + xpos[4]] = true;
						else
							bAllRemovable = false;

						int index = xpos[0] + m_nWidth * ypos[0];
						if (bAllRemovable)
							pSuccessMap[index] = true;
						else
							pSuccessMap[index] = false;
					}
					else
					{
						int index = xpos[0] + m_nWidth * ypos[0];
						pSuccessMap[index] = false;
					}
				}
			}
		}
	}

	//LOD ������ ����� �̿��ؼ� ���ؽ� �ɷ�����
	int nCountRemovable = 0;
	int nCountVertices = 0;
	int nCountOutside = 0;
	for (int j = 0; j < m_nHeight; ++j)
	{
		for (int i = 0; i < m_nWidth; ++i)
		{
			int index = m_nWidth * j + i;
			if (m_pIsInsideRadiusMap[index])
			{
				if (m_pRemovable[index])
				{
					++nCountRemovable;
				}
				else
				{
					++nCountVertices;
				}
			}
			else
			{
				++nCountOutside;
			}
		}
	}
	if ((nCountVertices + nCountRemovable + nCountOutside) != m_nResolution)
	{
		fprintf(stderr, "the number of vertices does not match: %d, %d, %d \n", nCountVertices, nCountRemovable, nCountOutside);
		exit(ERROR_INVALID_GEOMETRY);
	}

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	//�̺κ� ���߿� ��������. ���� ��� ������
	m_vertices.reserve(nCountVertices + 100);

	int nHalf = m_nWidth >> 1;

	for (int j = 0; j < m_nHeight; ++j)
	{
		for (int i = 0; i < m_nWidth; ++i)
		{
			int index = m_nWidth * j + i;
			if (m_pIsInsideRadiusMap[index] && !m_pRemovable[index])
			{
				m_vertices.push_back(del_point2d_t(i - nHalf, j - nHalf));
			}
		}
	}
}

void modelGen::computeLODRear()
{
	bool* pSuccessMap = new bool[m_nResolution];
	m_pRemovableRear = new bool[m_nResolution];

	for (int i = 0; i < m_nResolution; ++i)
	{
		pSuccessMap[i] = true;
		m_pRemovableRear[i] = false;
	}

	auto isFourPartsFlat = [=](const std::vector<int>& xpos, const std::vector<int>& ypos) {
		return (pSuccessMap[xpos[0] + m_nWidth * ypos[0]] && pSuccessMap[xpos[1] + m_nWidth * ypos[1]] &&
			pSuccessMap[xpos[3] + m_nWidth * ypos[3]] && pSuccessMap[xpos[4] + m_nWidth * ypos[4]]);
	};

	auto isFourPartsOutside = [=](const std::vector<int>& xpos, const std::vector<int>& ypos) {
		return (!m_pIsInsideRadiusMap[xpos[0] + m_nWidth * ypos[0]] && !m_pIsInsideRadiusMap[xpos[1] + m_nWidth * ypos[1]] &&
			!m_pIsInsideRadiusMap[xpos[3] + m_nWidth * ypos[3]] && !m_pIsInsideRadiusMap[xpos[4] + m_nWidth * ypos[4]]);
	};

	for (int level = 2; level < 512; level *= 2)
	{
		int halflevel = level >> 1;

		for (int jbegin = 0; jbegin < (m_nHeight - 1); jbegin += level)
		{
			for (int ibegin = 0; ibegin < (m_nWidth - 1); ibegin += level)
			{
				std::vector<int> xpos;	//9���� ���ؽ� x��ǥ
				std::vector<int> ypos;	//9���� ���ؽ� y��ǥ
				for (int j = jbegin; j <= (jbegin + level); j += halflevel)
				{
					for (int i = ibegin; i <= (ibegin + level); i += halflevel)
					{
						xpos.push_back(i);
						ypos.push_back(j);
					}
				}

				//���� 4���� ������ ���ؽ��� ��� radius�ٱ��̶�� 
				if (isFourPartsOutside(xpos, ypos))
				{

				}
				else
				{
					if (isFourPartsFlat(xpos, ypos))	//4���� ������Ʈ�� ���� �������� faltten �Ǿ��ٸ� ture, ���� �߰��� flatten ����
					{
						bool bAllRemovable = true;

						//��ܰ� ���� ������ ���� �˻�� ���� �������� �ߺ��ǹǷ� �̹� �˻��ߴٸ� pass
						if (m_pRemovableRear[m_nWidth * ypos[1] + xpos[1]] || isVertexRemovableRear(xpos[0], ypos[0], xpos[2], ypos[2], xpos[1], ypos[1]))
							m_pRemovableRear[m_nWidth * ypos[1] + xpos[1]] = true;
						else
							bAllRemovable = false;

						//��ܰ� ���� ������ ���� �˻�� ���� �������� �ߺ��ǹǷ� �̹� �˻��ߴٸ� pass
						if (m_pRemovableRear[m_nWidth * ypos[3] + xpos[3]] || isVertexRemovableRear(xpos[0], ypos[0], xpos[6], ypos[6], xpos[3], ypos[3]))
							m_pRemovableRear[m_nWidth * ypos[3] + xpos[3]] = true;
						else
							bAllRemovable = false;

						if (isVertexRemovableRear(xpos[2], ypos[2], xpos[8], ypos[8], xpos[5], ypos[5]))
							m_pRemovableRear[m_nWidth * ypos[5] + xpos[5]] = true;
						else
							bAllRemovable = false;

						if (isVertexRemovableRear(xpos[6], ypos[6], xpos[8], ypos[8], xpos[7], ypos[7]))
							m_pRemovableRear[m_nWidth * ypos[7] + xpos[7]] = true;
						else
							bAllRemovable = false;

						if (isVertexRemovableRear(xpos[0], ypos[0], xpos[2], ypos[2], xpos[6], ypos[6], xpos[8], ypos[8], xpos[4], ypos[4]))
							m_pRemovableRear[m_nWidth * ypos[4] + xpos[4]] = true;
						else
							bAllRemovable = false;

						int index = xpos[0] + m_nWidth * ypos[0];
						if (bAllRemovable)
							pSuccessMap[index] = true;
						else
							pSuccessMap[index] = false;
					}
					else
					{
						int index = xpos[0] + m_nWidth * ypos[0];
						pSuccessMap[index] = false;
					}
				}
			}
		}
	}

	//LOD ������ ����� �̿��ؼ� ���ؽ� �ɷ�����
	int nCountRemovable = 0;
	int nCountVertices = 0;
	int nCountOutside = 0;
	for (int j = 0; j < m_nHeight; ++j)
	{
		for (int i = 0; i < m_nWidth; ++i)
		{
			int index = m_nWidth * j + i;
			if (m_pIsInsideRadiusMap[index])
			{
				if (m_pRemovableRear[index])
				{
					++nCountRemovable;
				}
				else
				{
					++nCountVertices;
				}
			}
			else
			{
				++nCountOutside;
			}
		}
	}
	if ((nCountVertices + nCountRemovable + nCountOutside) != m_nResolution)
	{
		fprintf(stderr, "the number of vertices does not match: %d, %d, %d \n", nCountVertices, nCountRemovable, nCountOutside);
		exit(ERROR_INVALID_GEOMETRY);
	}

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	//�̺κ� ���߿� ��������. ���� ��� ������
	m_verticesRear.reserve(nCountVertices + 100);

	int nHalf = m_nWidth >> 1;

	for (int j = 0; j < m_nHeight; ++j)
	{
		for (int i = 0; i < m_nWidth; ++i)
		{
			int index = m_nWidth * j + i;
			if (m_pIsInsideRadiusMap[index] && !m_pRemovableRear[index])
			{
				m_verticesRear.push_back(del_point2d_t(i - nHalf, j - nHalf));
			}
		}
	}
}

void modelGen::computeHeightMap()
{
	//�� �����鵵 �����ϰ� �ٲ���
	float fHeightFactor = 7.5f;
	//float fDistanceFactor = 0.005f;
	float fDistanceFactor = 0.00125f;
	float fFaceFactor = 1.f;
	float fHairFactor = 1.f;
	float fNoseFactor = 10.f;
	//float fEtcFactor = 0.75f;
	float fEtcFactor = 3.25f;

	float fHairEdgeFactor = 1.25f;
	float fEyeEdgeFactor = 1.0f;
	float fLipEdgeFactor = 0.125f;
	

	m_pHeightMap = new float[m_nResolution];

	for (int j = 0; j < m_nHeight; ++j)
	{
		for (int i = 0; i < m_nWidth; ++i)
		{
			float height;
			int index = m_nWidth * (m_nHeight - j - 1) + i;

			if (m_pTextMap && m_pTextMap[index])
				height = m_fTextHeight;
			else
			{
				if (m_pFacialComponentMap[index] == facialComponent::background)
					height = 0.f;

				//�Ӹ�ī��, ����
				else if ((m_pFacialComponentMap[index] == facialComponent::hair) || (m_pFacialComponentMap[index] == facialComponent::hat))
				{
					height = fHeightFactor * fFaceFactor * atan(fDistanceFactor * m_pDistanceMapHead[index]) * 1.5f;
					height += (fHeightFactor * fHairFactor * atan(fDistanceFactor * m_pDistanceMapEyebrow[index])) * 1.5f;
					height -= (fHairEdgeFactor * m_pDistanceMapEdge[index]);
				}

				// ��
				else if (m_pFacialComponentMap[index] == facialComponent::cloth)
					height = fHeightFactor * fFaceFactor * atan(fDistanceFactor * 0.5f * m_pDistanceMapHead[index]) * 1.3f;
				
				// ��, �����
				else if ((m_pFacialComponentMap[index] == facialComponent::neck) || (m_pFacialComponentMap[index] == facialComponent::necklace))
					height = fHeightFactor * 0.5f * atan(fDistanceFactor * m_pDistanceMapHead[index]) * 1.0f;

				else
				{
					height = fHeightFactor * fFaceFactor * atan(fDistanceFactor * m_pDistanceMapHead[index]) * 1.8f;

					//����
					if ((m_pFacialComponentMap[index] == facialComponent::eyebrowL) || (m_pFacialComponentMap[index] == facialComponent::eyebrowR))
					{
						height += (fEtcFactor * fHeightFactor * atan(fDistanceFactor * m_pDistanceMapEyebrow[index])) * 2.5;
						//height -= (0.25 * m_pRelativeBrightness[index]);
					}
					//�Ȱ�
					else if(m_pFacialComponentMap[index] == facialComponent::eyeGlasses)
						height += 1.5f;
					//��
					else if ((m_pFacialComponentMap[index] == facialComponent::eyeL) || (m_pFacialComponentMap[index] == facialComponent::eyeR))
						height -= (fEyeEdgeFactor * m_pDistanceMapEdge[index]);

					// ��
					//else if ((m_pFacialComponentMap[index] == facialComponent::earL) || (m_pFacialComponentMap[index] == facialComponent::earR))
					//	height += (0.15 * m_pRelativeBrightness[index]) * 1.8f;

					// �Լ� ��, �Ʒ�
					else if ((m_pFacialComponentMap[index] == facialComponent::lipU) || (m_pFacialComponentMap[index] == facialComponent::lipD))
					{
						height += (fEtcFactor * fHeightFactor * atan(fDistanceFactor * m_pDistanceMapEyebrow[index])) * 1.8f;
						height -= (fLipEdgeFactor * m_pDistanceMapEdge[index]);
					}

					// ��
					else if (m_pFacialComponentMap[index] == facialComponent::nose)
					{
						float distNose = m_pDistanceMapNoseDorsum[index];
						float distNoseBoundary = m_pDistanceMapNose[index];
						float noseDistRate = (distNose / (distNose + distNoseBoundary)) * 1.570796f;
						//float noseDistRate = (distNose / (distNose + distNoseBoundary));
						float noseHeight = float(m_pIndexMapNoseDorsum[index]) / m_seedsNoseDorsum.size();
						height += (fNoseFactor * noseHeight * gauss(noseDistRate)) * 1.8f;
						//height += (0.05f * m_pRelativeBrightness[index]) * 1.8f;
					}
				}
				//height += (fGradient * m_pRelativeBrightness[index]);
			}
			int index2 = m_nWidth * j + i;
			if (height < 0.f)
				height = 0.f;

			m_pHeightMap[index2] = height;
		}
	}
}

void modelGen::computeHeightMapEdgeOnly()
{
	//�� �����鵵 �����ϰ� �ٲ���
	float fHeightFactor = 2.5f;
	float fDistanceFactor = 0.5f;
	
	m_originalImageSmooth1 = m_originalImageGrayscale.clone();
	cv::GaussianBlur(m_originalImageGrayscale, m_originalImageSmooth1, cv::Size(9, 9), 0, 0);

	m_originalImageSmooth2 = m_originalImageGrayscale.clone();
	cv::GaussianBlur(m_originalImageGrayscale, m_originalImageSmooth2, cv::Size(5, 5), 0, 0);

	m_relativeBrightnessImage = m_originalImageGrayscale.clone();
	
	uchar* pPixel0 = m_relativeBrightnessImage.data;
	uchar* pPixel1 = m_originalImageSmooth1.data;
	uchar* pPixel2 = m_originalImageSmooth2.data;
	
	for (int index = 0; index < m_nResolution; ++index)
	{
		int temp = *pPixel2++ - *pPixel1++;
		if (temp < 0)
			temp = -temp;
		*pPixel0++ = uchar(temp);
	}
	cv::imwrite("res00.png", m_relativeBrightnessImage);
	
	m_pHeightMap = new float[m_nResolution];

	for (int j = 0; j < m_nHeight; ++j)
	{
		for (int i = 0; i < m_nWidth; ++i)
		{
			float height;
			int new_j = m_nHeight - j - 1;
			int index = m_nWidth * new_j + i;
			int index2 = m_nWidth * j + i;

			if (m_pTextMap && m_pTextMap[index])
				height = m_fTextHeight;
			else
			{
				int temp = 0;
				if (m_relativeBrightnessImage.at<uchar>(new_j, i) >= 3)
					temp = m_relativeBrightnessImage.at<uchar>(new_j, i);

				int temp2 = (m_originalImageSmooth2.at<uchar>(new_j, i) - 128) * -0.5f;
				if (temp2 < 0.f)
					temp2 *= 0.25f;

				temp += temp2;

				height = -(temp * 0.3f);
				//height = -(m_relativeBrightnessImage.at<uchar>(new_j, i) * 0.1f);
				//height = -(m_originalImageGrayscale.at<uchar>(new_j, i) * 0.1f);
				
				if (m_edgeImage.at<uchar>(new_j, i) < 244)
					height = -fHeightFactor * atan(fDistanceFactor * m_pDistanceMapEdge[index2]);
				
			}
			
			if (height < -10.f)
				height = -10.f;

			m_pHeightMap[index2] = height;
		}
	}
}

void modelGen::computeHeightMapRear()
{
	//�� �����鵵 �����ϰ� �ٲ���
	float fHeightFactor = 7.5f;
	
	m_pHeightMapRear = new float[m_nResolution];

	for (int j = 0; j < m_nHeight; ++j)
	{
		for (int i = 0; i < m_nWidth; ++i)
		{
			float height;
			int reverseJ = m_nHeight - j - 1;

			if (m_rearImageGrayscale.at<uchar>(j, i) < 128)
				height = -5.f;
			else
				height = 0.f;
				
			int index2 = m_nWidth * reverseJ + i;
			
			m_pHeightMapRear[index2] = height;
		}
	}
}

void modelGen::tessellate()
{
	delaunay2d_t* pDelaunay = delaunay2d_from(m_vertices.data(), m_vertices.size());
	m_pFaces = tri_delaunay2d_from(pDelaunay);

	int nNumTriangles = m_pFaces->num_triangles;
	int nNumVertex = m_pFaces->num_points;
}

void modelGen::tessellateRear()
{
	delaunay2d_t* pDelaunay = delaunay2d_from(m_verticesRear.data(), m_verticesRear.size());
	m_pFacesRear = tri_delaunay2d_from(pDelaunay);

}

void modelGen::generateBoundaryCircle()
{
	int nExpandedRaius = (m_nWidth >> 1) + 1;
	
	float fDegree = 2.5f;						//5�� ���� (degree)
	float fAngle = 3.141592f / 180 * fDegree;	//5�� ���� (radian)
	int nCount = 360 / fDegree;
	m_nBoundaryVerticesNum = nCount;

	float x = 0.f;
	float y = nExpandedRaius;

	m_vertices.push_back(del_point2d_t(x, y));

	for (int i = 1; i < nCount; ++i)
	{
		float newX = x * cos(i * fAngle) - y * sin(i * fAngle);
		float newY = x * sin(i * fAngle) + y * cos(i * fAngle);
		m_vertices.push_back(del_point2d_t(newX, newY));
	}
}

void modelGen::generateBoundaryCircleRear()
{
	int nExpandedRaius = (m_nWidth >> 1) + 1;

	float fDegree = 2.5f;						//5�� ���� (degree)
	float fAngle = 3.141592f / 180 * fDegree;	//5�� ���� (radian)
	
	float x = 0.f;
	float y = nExpandedRaius;

	m_verticesRear.push_back(del_point2d_t(x, y));

	for (int i = 1; i < m_nBoundaryVerticesNum; ++i)
	{
		float newX = x * cos(i * fAngle) - y * sin(i * fAngle);
		float newY = x * sin(i * fAngle) + y * cos(i * fAngle);
		m_verticesRear.push_back(del_point2d_t(newX, newY));
	}
}

inline float modelGen::gauss(float x) {
	const float sigma = 0.75f;
	x *= 1.5f;
	return (1.f / (sigma * sqrt(2.f * 3.141592f))) * exp(-0.5f * pow((x) / sigma, 2.f));
}

bool modelGen::isVertexRemovable(int x1, int y1, int x2, int y2, int cx, int cy)
{
	if ((m_pIsInsideRadiusMap[m_nWidth * y1 + x1] == false) || (m_pIsInsideRadiusMap[m_nWidth * y2 + x2] == false) || (m_pIsInsideRadiusMap[m_nWidth * cy + cx] == false))
		return false;
	
	float fHeight1 = m_pHeightMap[m_nWidth * y1 + x1];
	float fHeight2 = m_pHeightMap[m_nWidth * y2 + x2];
	float fHeightc = m_pHeightMap[m_nWidth * cy + cx];

	if (abs(((fHeight1 + fHeight2) * 0.5f) - fHeightc) <= m_fLodError)
		return true;
	else
		return false;
}

bool modelGen::isVertexRemovableRear(int x1, int y1, int x2, int y2, int cx, int cy)
{
	if ((m_pIsInsideRadiusMap[m_nWidth * y1 + x1] == false) || (m_pIsInsideRadiusMap[m_nWidth * y2 + x2] == false) || (m_pIsInsideRadiusMap[m_nWidth * cy + cx] == false))
		return false;

	float fHeight1 = m_pHeightMapRear[m_nWidth * y1 + x1];
	float fHeight2 = m_pHeightMapRear[m_nWidth * y2 + x2];
	float fHeightc = m_pHeightMapRear[m_nWidth * cy + cx];

	if (abs(((fHeight1 + fHeight2) * 0.5f) - fHeightc) <= m_fLodErrorRear)
		return true;
	else
		return false;
}

inline bool modelGen::isVertexRemovable(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4, int cx, int cy)
{
	return (isVertexRemovable(x1, y1, x4, y4, cx, cy) && isVertexRemovable(x2, y2, x3, y3, cx, cy));
}

inline bool modelGen::isVertexRemovableRear(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4, int cx, int cy)
{
	return (isVertexRemovableRear(x1, y1, x4, y4, cx, cy) && isVertexRemovableRear(x2, y2, x3, y3, cx, cy));
}