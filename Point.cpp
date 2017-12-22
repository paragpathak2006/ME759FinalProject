#pragma once
#include<iostream>
//#include<conio.h>
//#include<conio.h>
#include<math.h>
#define PI 3.14159265

using namespace std;
class Point
{
public:
	float x,y,z,w;
	__host__ __device__ Point();
	__host__ __device__ Point(float X,float Y,float Z);
	__host__ __device__ Point(float X,float Y,float Z,float W);
	Point cart();
	void print(FILE *fp);
	void print();
	__host__ __device__ friend float d(Point P1,Point P2);
	__host__ __device__ Point operator+(Point P);
	__host__ __device__ Point operator-(Point P);
	__host__ __device__ Point operator*(float a);
	__host__ __device__ float operator*(Point a);
	__host__ __device__ Point operator%(Point b);
	__host__ __device__ Point operator/(float b);
	__host__ __device__ friend Point operator*(float b,Point a);
	__host__ __device__ friend float norm(Point a);
	__host__ __device__ friend Point R(Point a,Point u,float b);
};

float norm(Point U)
{
	float u = U.x,v = U.y,w = U.z;	
	return 	sqrt(u*u + v*v + w*w);
}

Point R(Point P,Point U,float deg)
{
	float rad,C,S,UX,U2;
	float x,y,z;
	float u,v,w;
	float Px,Py,Pz;
	
	rad = deg*PI/180.0;	C = cos(rad);	S = sin(rad);
	
	x = P.x;	y = P.y;	z = P.z;	
	u = U.x;	v = U.y;	w = U.z;	
	
	UX = P*U; 	U2 = (u*u + v*v + w*w);
	
	Px = (u*UX*(1-C) + U2*x*C + sqrt(U2)*(-w*y+v*z)*S)/U2;
	Py = (v*UX*(1-C) + U2*y*C + sqrt(U2)*(w*x-u*z)*S)/U2;
	Pz = (w*UX*(1-C) + U2*z*C + sqrt(U2)*(-v*x+u*y)*S)/U2;
	return Point(Px,Py,Pz);
}


float d(Point P1,Point P2)
{
	return sqrt((P1.x-P2.x)*(P1.x-P2.x) + (P1.y-P2.y)*(P1.y-P2.y) +(P1.z-P2.z)*(P1.z-P2.z));
}
Point Point::cart()
{
//	return Point(x/w,y/w,z/w,w);
	return Point(x/w,y/w,z/w,1.0);
}
void Point::print(FILE *fp)
{
	Point P = cart();
//	fprintf(fp,"%f %f %f %f\t",P.x,P.y,P.z,P.w);
//	fprintf(fp,"%f %f %f %f\n",P.x,P.y,P.z,P.w);
	fprintf(fp,"%f %f %f %f\n",P.x/P.w,P.y/P.w,P.z/P.w,1.0);

}
void Point::print()
{
	Point P = cart();
//	fprintf(fp,"%f %f %f %f\t",P.x,P.y,P.z,P.w);
	printf("//%f %f %f\n",P.x,P.y,P.z);
}

Point::Point()
{
	x=0.0;y=0.0;z=0.0;w=1.0;
}
Point::Point(float X,float Y,float Z)
{
	x=X;y=Y;z=Z;w=1;	
}

Point::Point(float X,float Y,float Z,float W)
{
	x=X;y=Y;z=Z;w=W;	
}

Point Point::operator+(Point P)
{
	return Point(x+P.x,y+P.y,z+P.z,w+P.w);
}
Point Point::operator-(Point P)
{
	return Point(x-P.x,y-P.y,z-P.z,w-P.w);
}
Point Point::operator*(float a)
{
	return Point(x*a,y*a,z*a,w*a);
}
//-----------**************---------------operator descriptions----------*************-----------------
float Point::operator*(Point b)
{
	return x*b.x+y*b.y+z*b.z;
}
//-----------**************---------------operator descriptions----------*************-----------------
Point Point::operator%(Point b)
{
	return Point(y*b.z - z*b.y,z*b.x - x*b.z,x*b.y - y*b.x);			  
}


//-----------**************---------------operator descriptions----------*************-----------------
Point Point::operator/(float b)
{
	return Point(x/b,y/b,z/b,w/b);;			  
}
//-----------**************---------------operator descriptions----------*************-----------------
Point operator*(float b,Point a){return a*b;}
