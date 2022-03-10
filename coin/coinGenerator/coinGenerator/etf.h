#pragma once
#include "shared.h"

class vec 
{
public:
	double tx = 0.f;
	double ty = 0.f;
	double mag = 0.f;

	vec() = default;
	vec(double i, double j) : tx(i), ty(j) { mag = sqrt(i * i + j * j); };
	vec operator+(const vec& v) const;
	vec operator+(const double d)const;
	vec operator+=(const vec& v) ;
	vec operator+=(const double d);
	vec operator-() const;
	vec operator-(const vec& v) const;
	vec operator-=(const vec& v);
	vec operator-(const double d) const;
	vec operator-=(const double d);
	vec operator*(const double d) const;
	vec operator/(const double d) const;

	void makeUnit();
};

class ETF
{
	cv::Mat inputImage_;
	cv::Mat resultImage_;
	int width;
	int height;

	double max_grad = -1.f;
	vec** pVectors = nullptr;

	void clean();
	void copyVectorsFrom(const ETF& e);
	void makeGaussian(double sigma, std::vector<double>& gaussian);
	void GetDirectionalDoG(cv::Mat& dogImage, std::vector<double>& gaussian1, std::vector<double>& gaussian2, double tau);
	void GetFlowDoG(cv::Mat& dogImage, cv::Mat& tempImage, std::vector<double>& gaussian);

public:
	ETF() = default;
	ETF(const ETF& e);
	~ETF() { clean(); };

	void setImage(const cv::Mat& image);
	void smooth(int radius, int iteration);
	void computeFDoG(double sigma, double sigma3, double tau);
	cv::Mat& getFDoGImage() { return resultImage_; };
};