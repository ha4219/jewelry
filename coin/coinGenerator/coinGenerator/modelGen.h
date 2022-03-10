#pragma once
#include "shared.h"
#include <stdio.h>
#include "delaunay.h"
#include "etf.h"

enum coinShape
{
	circle, rectangle, diamond
};

enum facialComponent
{
	background = 0,
	skin = 1,
	eyebrowL = 2,
	eyebrowR = 3,
	eyeL = 4,
	eyeR = 5,
	eyeGlasses = 6,
	earL = 7,
	earR = 8,
	earRing = 9,
	nose = 10,
	mouth = 11,
	lipU = 12,
	lipD = 13,
	neck = 14,
	necklace = 15,
	cloth = 16,
	hair = 17,
	hat = 18,
	undefined
};

class dwSeed
{
	int x;
	int y;
	//float maxDist = 0.f;
public:
	dwSeed(int x, int y) {
		this->x = x; this->y = y;
	}
	int getX() const { return x; }
	int getY() const { return y; }
	/*
	double getMaxDist() const { return maxDist; }
	float calcAndUpdateDistance(int x, int y) {
		float dist = sqrt(pow(this->x - x, 2) + pow(this->y - y, 2));
		if (dist > maxDist)
			maxDist = dist;
		return dist;
	}*/
};

class dwVec3D
{
public:
	float x = 0.f;
	float y = 0.f;
	float z = 0.f;
	dwVec3D() {}
	dwVec3D(float x, float y, float z) {
		this->x = x;
		this->y = y;
		this->z = z;
	}
	dwVec3D crossProduct(const dwVec3D& vec) {
		dwVec3D ret;
		ret.x = y * vec.z - z * vec.y;
		ret.y = z * vec.x - x * vec.z;
		ret.z = x * vec.y - y * vec.x;
		return ret;
	}
	inline void makeUnitVec();
	inline float getLength();
};

class modelGen
{
	coinShape m_shape = coinShape::circle;

	bool m_bEdgeOnlyMode = false;
	cv::Mat m_originalImageGrayscale;
	cv::Mat m_rearImageGrayscale;
	cv::Mat m_edgeImage;
	cv::Mat m_originalImageSmooth1;
	cv::Mat m_originalImageSmooth2;
	cv::Mat m_relativeBrightnessImage;

	ETF m_etf;
	const int m_nEyeExpansion = 10;
	const int m_nFilterWidth = 1025;
	const int m_nFilterCentre = 512;

	int m_nWidth;
	int m_nHeight;
	int m_nResolution;
	int m_nMaxDiagonal;

	int	  m_nModelSize;					//�̰� ���߿� �����ؾ� ��: ���� ��¥ ũ��
	float m_fTextHeight = 8.f;			//�̰� ���߿� �����ؾ� ��: �ؽ�Ʈ�� ����
	const float m_fLodError = 0.25f;	//�̰� ���߿� �����ؾ� ��: LOD �������� ���̿��� ��� ����
	const float m_fLodErrorRear = 0.125f;	//�̰� ���߿� �����ؾ� ��: LOD �������� ���̿��� ��� ����
	//const float m_fLodError = 0.f;	//�̰� ���߿� �����ؾ� ��: LOD �������� ���̿��� ��� ����

	int m_nBoundaryVerticesNum;

	std::vector<dwSeed> m_seedsEdges;
	std::vector<dwSeed> m_seedsBoundaryHead;
	std::vector<dwSeed> m_seedsBoundaryEyebrow;
	std::vector<dwSeed> m_seedsBoundaryNose;
	std::vector<dwSeed> m_seedsNoseDorsum;
	dwVec3D m_noseDorsumFeatures[4];
	std::vector<del_point2d_t> m_vertices;
	std::vector<del_point2d_t> m_verticesRear;
	tri_delaunay2d_t* m_pFaces = NULL;
	tri_delaunay2d_t* m_pFacesRear = NULL;

	unsigned char* m_pFacialComponentMap = NULL;
	unsigned char* m_pFacialComponentBoundaryMap = NULL;

	float* m_pDistanceFilter = NULL;
	float* m_pDistanceMapEdge = NULL;
	float* m_pDistanceMapHead = NULL;
	float* m_pDistanceMapEyebrow = NULL;		//����, �Լ�, ��� �ٿ������ ���� ���Ͻ���
	float* m_pDistanceMapNose = NULL;			//�� ���� �ٿ������ ���� ���Ͻ���
	float* m_pDistanceMapNoseDorsum = NULL;		//�� ���� ���Ͻ���
	int*   m_pIndexMapNoseDorsum = NULL;
	bool*  m_pTextMap = NULL;
	float* m_pHeightMap = NULL;
	float* m_pHeightMapRear = NULL;
	bool*  m_pIsInsideRadiusMap = NULL;
	bool*  m_pRemovable = NULL;
	bool*  m_pRemovableRear = NULL;
	

	void expandEyeRegionVer1(int nRadius);
	void expandEyeRegionVer2();
	void initDistanceFilter();
	void generateFacialComponentBoundary();
	void computeDistanceMap();
	void computeDistanceMapEdgeOnly();
	void computeHeightMap();
	void computeHeightMapEdgeOnly();
	void computeHeightMapRear();
	void computeIfIsInsideRadius();
	void computeLOD();
	void computeLODRear();
	void tessellate();
	void tessellateRear();
	void generateBoundaryCircle();
	void generateBoundaryCircleRear();
	void exportModel(std::string& strFileName, float fProductRadius);

	unsigned int getVerticesNum() const { return m_pFaces->num_points; };
	unsigned int getFacesNum() const { return m_pFaces->num_triangles; };
	unsigned int getVerticesNumRear() const { return m_pFacesRear->num_points; };
	unsigned int getFacesNumRear() const { return m_pFacesRear->num_triangles; };

	inline bool isVertexRemovable(int x1, int y1, int x2, int y2, int cx, int cy);
	inline bool isVertexRemovable(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4, int cx, int cy);
	inline bool isVertexRemovableRear(int x1, int y1, int x2, int y2, int cx, int cy);
	inline bool isVertexRemovableRear(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4, int cx, int cy);
	inline float gauss(float x);

public:
	modelGen() {
		m_seedsBoundaryHead.reserve(4096);
		m_seedsBoundaryEyebrow.reserve(16384);
		m_seedsBoundaryEyebrow.reserve(4096);
	};
	void setImage(std::string& strFileName);
	void setImageRear(std::string& strFileName);
	void setMask(std::string& strFileName);
	void setNoseDorsum(std::string& strFileName);
	void setTextImage(std::string& strFileName);

	void generateNoseDorsumLine();
	void preComputeHeightMap();

	void generate3Dmodel(std::string& strFileName, float fProductRadius);
	void setEdgeOnlyMode();
};
