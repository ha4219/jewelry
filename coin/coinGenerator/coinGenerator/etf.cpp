#include "etf.h"
#include <opencv2/imgproc/imgproc.hpp>

vec vec::operator+(const double d) const
{
	vec ret;
	ret.tx = tx + d;
	ret.ty = ty + d;
	ret.mag = sqrt((ret.tx * ret.tx) + (ret.ty * ret.ty));
	return ret;
}
vec vec::operator+(const vec& v) const
{
	vec ret;
	ret.tx = tx + v.tx;
	ret.ty = ty + v.ty;
	ret.mag = sqrt((ret.tx * ret.tx) + (ret.ty * ret.ty));
	return ret;
}
vec vec::operator+=(const vec& v)
{
	(*this) = ((*this) + v);
	return (*this);
}
vec vec::operator+=(const double d)
{
	(*this) = ((*this) + d);
	return (*this);
}
vec vec::operator-() const
{
	vec ret;
	ret.tx = -tx;
	ret.ty = -ty;
	return ret;
}
vec vec::operator-(const vec& v) const
{
	vec ret;
	ret.tx = tx - v.tx;
	ret.ty = ty - v.ty;
	ret.mag = sqrt((ret.tx * ret.tx) + (ret.ty * ret.ty));
	return ret;
}
vec vec::operator-(const double d) const
{
	vec ret;
	ret.tx = tx - d;
	ret.ty = ty - d;
	ret.mag = sqrt((ret.tx * ret.tx) + (ret.ty * ret.ty));
	return ret;
}
vec vec::operator-=(const vec& v)
{
	(*this) = ((*this) - v);
	return (*this);
}
vec vec::operator-=(const double d) 
{
	(*this) = ((*this) - d);
	return (*this);
}
vec vec::operator*(const double d) const
{
	vec ret;
	ret.tx = tx * d;
	ret.ty = ty * d;
	ret.mag = mag * d;
	return ret;
}
vec vec::operator/(const double d) const
{
	vec ret;
	ret.tx = tx / d;
	ret.ty = ty / d;
	ret.mag = mag * d;
	return ret;
}

void vec::makeUnit()
{
	double magTemp = sqrt(tx * tx + ty * ty);
	if (mag != 0.0)
	{
		tx /= magTemp;
		ty /= magTemp;
		mag = magTemp;
	}
}

ETF::ETF(const ETF& e)
{
	inputImage_ = e.inputImage_.clone();
	resultImage_ = e.resultImage_.clone();

	width = e.width;
	height = e.height;
	max_grad = e.max_grad;

	pVectors = new vec * [height];
	for (int i = 0; i < height; ++i)
	{
		pVectors[i] = new vec[width];
	}

	for (int j = 0; j < height; ++j)
	{
		for (int i = 0; i < width; ++i)
		{
			pVectors[j][i] = e.pVectors[j][i];
		}
	}

}

void ETF::copyVectorsFrom(const ETF& e)
{
	for (int j = 0; j < height; ++j)
	{
		for (int i = 0; i < width; ++i)
		{
			pVectors[j][i] = e.pVectors[j][i];
		}
	}
}

void ETF::setImage(const cv::Mat& image)
{
	inputImage_ = image.clone();
	resultImage_ = image.clone();
	width = inputImage_.cols;
	height = inputImage_.rows;

	clean();
	pVectors = new vec*[height];
	for (int i = 0; i < height; ++i)
	{
		pVectors[i] = new vec[width];
	}

	const double MAX_VAL = 1020.;
	vec v;
	
	uchar* pPixel;
	for (int j = 1; j < height - 1; ++j)
	{
		for (int i = 1; i < width - 1; ++i)
		{
			pVectors[j][i].tx = (inputImage_.at<uchar>(j - 1, i + 1) + 2.f * inputImage_.at<uchar>(j, i + 1) + inputImage_.at<uchar>(j + 1, i + 1)
				- inputImage_.at<uchar>(j - 1, i - 1) - 2.f * inputImage_.at<uchar>(j, i - 1) - inputImage_.at<uchar>(j + 1, i - 1)) / MAX_VAL;
			pVectors[j][i].ty = (inputImage_.at<uchar>(j + 1, i + 1) + 2.f * inputImage_.at<uchar>(j + 1, i) + inputImage_.at<uchar>(j + 1, i + 1)
				- inputImage_.at<uchar>(j - 1, i - 1) - 2.f * inputImage_.at<uchar>(j - 1, i) - inputImage_.at<uchar>(j - 1, i + 1)) / MAX_VAL;
			
			v = pVectors[j][i];
			pVectors[j][i].tx = -v.ty;
			pVectors[j][i].ty = v.tx;
			pVectors[j][i].mag = sqrt(pVectors[j][i].tx * pVectors[j][i].tx + pVectors[j][i].ty * pVectors[j][i].ty);
			
			if (pVectors[j][i].mag > max_grad) {
				max_grad = pVectors[j][i].mag;
			}
		}
	}

	for (int j = 1; j <= height - 2; ++j)
	{
		pVectors[j][0] = pVectors[j][1];
		pVectors[j][width - 1] = pVectors[j][width - 2];
	}

	for (int i = 1; i <= width - 2; ++i)
	{
		pVectors[0][i] = pVectors[1][i];
		pVectors[height - 1][i] = pVectors[height - 2][i];
	}

	pVectors[0][0] = (pVectors[0][1] + pVectors[1][0]) / 2;
	pVectors[0][width - 1] = (pVectors[0][width - 2] + pVectors[1][width - 1]) / 2;
	pVectors[height - 1][0] = (pVectors[height - 1][1] + pVectors[height - 2][0]) / 2;
	pVectors[height - 1][width - 1] = (pVectors[height - 1][width - 2] + pVectors[height - 2][width - 1]) / 2;

	for (int j = 0; j < height; ++j)
	{
		for (int i = 0; i < width; ++i)
		{
			pVectors[j][i].makeUnit();
		}
	}

}

void ETF::smooth(int radius, int iteration)
{
	ETF e2(*this);
	
	for (int k = 0; k < iteration; ++k)
	{
		// horizontal
		for (int j = 0; j < height; ++j)
		{
			for (int i = 0; i < width; ++i)
			{
				vec g;
				vec v = pVectors[j][i];
				for (int s = -radius; s <= radius; ++s)
				{
					int x = i + s; 
					int y = j;
					if (x > width - 1)
						x = width - 1;
					else if (x < 0) 
						x = 0;
					if (y > height - 1) 
						y = height - 1;
					else if (y < 0)
						y = 0;

					double mag_diff = pVectors[y][x].mag - pVectors[j][i].mag;
					vec w = pVectors[y][x];

					double factor = 1.0;
					double angle = v.tx * w.tx + v.ty * w.ty;
					if (angle < 0.0) {
						factor = -1.0;
					}
					double weight = mag_diff + 1;

					g += (pVectors[y][x] * weight * factor);
					g.makeUnit();
					e2.pVectors[j][i] = g;
				}
			}
		}
		copyVectorsFrom(e2);
		
		// vertical
		for (int j = 0; j < height; ++j)
		{
			for (int i = 0; i < width; ++i)
			{
				vec g;
				vec v = pVectors[j][i];
				for (int t = -radius; t <= radius; ++t)
				{
					int x = i;
					int y = j + t;
					if (x > width - 1)
						x = width - 1;
					else if (x < 0)
						x = 0;
					if (y > height - 1)
						y = height - 1;
					else if (y < 0)
						y = 0;

					double mag_diff = pVectors[y][x].mag - pVectors[j][i].mag;
					vec w = pVectors[y][x];

					double factor = 1.0;
					double angle = v.tx * w.tx + v.ty * w.ty;
					if (angle < 0.0) {
						factor = -1.0;
					}
					double weight = mag_diff + 1;

					g += (pVectors[y][x] * weight * factor);
					g.makeUnit();
					e2.pVectors[j][i] = g;
				}
			}
		}
		copyVectorsFrom(e2);
	}
}

void ETF::clean()
{
	if (pVectors)
	{
		for (int i = 0; i < height; ++i)
		{
			delete[] pVectors[i];
		}
		delete[] pVectors;
	}
}

void ETF::makeGaussian(double sigma, std::vector<double>& gaussian)
{
	auto gauss = [](double x, double mean, double sigma) {
		return (exp((-(x - mean) * (x - mean)) / (2 * sigma * sigma)) / sqrt(3.141592 * 2.0 * sigma * sigma));
	};

	double g;
	int i = 0;
	const double threshold = 0.001;
	do
	{
		g = gauss(i++, 0.0, sigma);
		gaussian.push_back(g);
	} while (g > threshold);
}

void ETF::computeFDoG(double sigma, double sigma3, double tau)
{
	std::vector<double> gaussian1;
	std::vector<double> gaussian2;
	std::vector<double> gaussian3;

	makeGaussian(sigma, gaussian1);
	makeGaussian(sigma * 1.6, gaussian2);
	makeGaussian(sigma3, gaussian3);

	int half_w1 = gaussian1.size() - 1;
	int half_w2 = gaussian2.size() - 1;
	int half_l = gaussian3.size() - 1;

	//cv::Mat dogImage = inputImage_.clone();
	cv::Mat dogImage = cv::Mat::zeros(cv::Size(width, height), CV_32FC1);
	cv::Mat tempImage = cv::Mat::zeros(cv::Size(width, height), CV_32FC1);

	GetDirectionalDoG(dogImage, gaussian1, gaussian2, tau);
	GetFlowDoG(dogImage, tempImage, gaussian3);

	uchar* pPixel1;
	float* pPixel2;
	for (int j = 0; j < height; ++j)
	{
		pPixel1 = resultImage_.ptr<uchar>(j);
		pPixel2 = tempImage.ptr<float>(j);
		for (int i = 0; i < width; ++i)
		{
			*pPixel1++ = uchar((*pPixel2++)*255.f + 0.5f);
		}
	}
	dogImage.release();
	tempImage.release();
}

void ETF::GetFlowDoG(cv::Mat& dogImage, cv::Mat& tempImage, std::vector<double>& gaussian)
{
	vec vt;
	int half_l = gaussian.size() - 1;

	double step_size = 1.0;

	for (int j = 0; j < height; ++j)
	{
		for (int i = 0; i < width; ++i)
		{
			double sum1 = 0.0;
			double w_sum1 = 0.0;
			double weight1 = 0.0;

			double val = dogImage.at<float>(j, i);
			weight1 = gaussian[0];
			sum1 = val * weight1;
			w_sum1 += weight1;

			vec d(i, j);
			int i_x = i; 
			int i_y = j;

			for (int k = 0; k < half_l; k++)
			{
				vt = pVectors[i_y][i_x];
				if (vt.tx == 0.0 && vt.ty == 0.0)
					break;
				vec xy = d;

				if (xy.tx > (double)width - 1 || xy.tx < 0.0 || xy.ty >(double)height - 1 || xy.ty < 0.0)
					break;

				vec xy1 = xy + 0.5;
				if (xy1.tx < 0)
					xy1.tx = 0;
				if (xy1.tx > width - 1)
					xy1.tx = width - 1;

				if (xy1.ty < 0)
					xy1.ty = 0;
				if (xy1.ty > height - 1)
					xy1.ty = height - 1;

				double val = dogImage.at<float>((int)xy1.ty, (int)xy1.tx);

				weight1 = gaussian[k];

				sum1 += val * weight1;
				w_sum1 += weight1;

				d += vt * step_size;
				
				i_x = int(d.tx + 0.5f);
				i_y = int(d.ty + 0.5f);

				if (d.tx < 0 || d.tx > width - 1 || d.ty < 0 || d.ty > height - 1)
					break;
			}

			d = vec(i, j);
			i_x = i; i_y = j;

			for (int k = 0; k < half_l; k++)
			{
				vt = -pVectors[i_y][i_x];
				if (vt.tx == 0.0 && vt.ty == 0.0)
					break;
				vec xy = d;

				if (xy.tx > (double)width - 1 || xy.tx < 0.0 || xy.ty >(double)height - 1 || xy.ty < 0.0)
					break;

				vec xy1 = xy + 0.5;
				if (xy1.tx < 0)
					xy1.tx = 0;
				if (xy1.tx > width - 1)
					xy1.tx = width - 1;

				if (xy1.ty < 0)
					xy1.ty = 0;
				if (xy1.ty > height - 1)
					xy1.ty = height - 1;

				double val = dogImage.at<float>((int)xy1.ty, (int)xy1.tx);

				weight1 = gaussian[k];

				sum1 += val * weight1;
				w_sum1 += weight1;

				d += vt * step_size;

				i_x = int(d.tx + 0.5f);
				i_y = int(d.ty + 0.5f);

				if (d.tx < 0 || d.tx > width - 1 || d.ty < 0 || d.ty > height - 1)
					break;
			}

			sum1 /= w_sum1;
			//////////////////////////////////////
			if (sum1 > 0)
				tempImage.at<float>(j,i) = 1.0;
			else
				tempImage.at<float>(j, i) = 1.0 + tanh(sum1);
		}
	}

}

void ETF::GetDirectionalDoG(cv::Mat& dogImage, std::vector<double>& gaussian1, std::vector<double>& gaussian2, double tau)
{
	int half_w1 = gaussian1.size() - 1;
	int half_w2 = gaussian2.size() - 1;

	for (int j = 0; j < height; ++j)
	{
		for (int i = 0; i < width; ++i)
		{
			double sum1 = 0.0;
			double sum2 = 0.0;
			double w_sum1 = 0.0;
			double w_sum2 = 0.0;
			
			vec vn;
			vn.tx = -(pVectors[j][i].ty);
			vn.ty = (pVectors[j][i].tx);

			if (vn.tx == 0.0 && vn.ty == 0.0)
			{
				sum1 = 255.0;
				sum2 = 255.0;
				dogImage.at<float>(j,i) = sum1 - tau * sum2;
				continue;
			}
			vec d(i, j);

			for (int s = -half_w2; s <= half_w2; s++)
			{
				vec xy = d + (vn * s);
				
				if (xy.tx > (double)width - 1 || xy.tx < 0.0 || xy.ty >(double)width - 1 || xy.ty < 0.0)
					continue;

				vec xy1 = xy + 0.5;
				if (xy1.tx < 0)
					xy1.tx = 0;
				if (xy1.tx > width - 1)
					xy1.tx = width - 1;

				if (xy1.ty < 0)
					xy1.ty = 0;
				if (xy1.ty > height - 1)
					xy1.ty = height - 1;

				double val = inputImage_.at<uchar>((int)xy1.ty, (int)xy1.tx);

				double weight1;
				double weight2;

				int dd = std::abs(s);
				if (dd > half_w1) 
					weight1 = 0.0;
				else
					weight1 = gaussian1[dd];

				sum1 += (val * weight1);
				w_sum1 += weight1;

				weight2 = gaussian2[dd];
				sum2 += (val * weight2);
				w_sum2 += weight2;
			}
			sum1 /= w_sum1;
			sum2 /= w_sum2;

			dogImage.at<float>(j, i) = sum1 - tau * sum2;
		}
	}
}