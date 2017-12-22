#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include <cuda.h>
#include <math.h>
#include "Point.cpp"
#include "stopwatch.hpp"
using namespace std;
__global__ void kernelFoo(float *CPx,float *CPy,float *CPz,float *u_knots,float *v_knots,float *opx,float *opy,float *opz,float *ofx,float *ofy,float *ofz)
{
	volatile __shared__ float u[6],v[6],uf,vf;
	volatile __shared__ float Px[4][4],P1x[4][4],Pux[4][4],Pvx[4][4];
	volatile __shared__ float Py[4][4],P1y[4][4],Puy[4][4],Pvy[4][4];
	volatile __shared__ float Pz[4][4],P1z[4][4],Puz[4][4],Pvz[4][4];
	volatile __shared__ int iff,jff;
	 	
	int i = threadIdx.x;
	int j = threadIdx.y;

	int ib = blockIdx.x;
	int jb = blockIdx.y;

	if(i==0&&j==0)
	{
		uf = ib*4.0/100.0 + 0.5*4.0/100.0; 
		vf = jb*4.0/100.0 + 0.5*4.0/100.0;

		for(int ii = 0;ii<9;ii++)   if(u_knots[ii]<=uf&&uf<u_knots[ii+1]) iff = ii;
		for(int jj = 0;jj<9;jj++)	if(v_knots[jj]<=vf&&vf<v_knots[jj+1]) jff = jj;

		for(int ii = 0;ii<6;ii++)   u[ii] = u_knots[ii + iff - 2];
		for(int jj = 0;jj<6;jj++)	v[jj] = v_knots[jj + jff - 2];
	}
			
	__syncthreads();	
	int ii = i + iff - 2;
	int jj = j + jff - 2;
	Px[i][j] = CPx[7*jj+ii];
	Py[i][j] = CPy[7*jj+ii];
	Pz[i][j] = CPz[7*jj+ii];

	__syncthreads();	
	for(int k = 0;	k<=2;	k++)
	{
		if(i<=2-k&&j<=2-k)
		{
			float uo = (uf-u[i+k])/(u[i+3]-u[i+k]);
			float vo = (vf-v[j+k])/(v[j+3]-v[j+k]);
			
			P1x[i][j] = Px[i][j]*(1.0-uo)*(1.0-vo) + Px[i+1][j]*uo*(1-vo) + Px[i][j+1]*vo*(1-uo)+Px[i+1][j+1]*uo*vo;
			P1y[i][j] = Py[i][j]*(1.0-uo)*(1.0-vo) + Py[i+1][j]*uo*(1-vo) + Py[i][j+1]*vo*(1-uo)+Py[i+1][j+1]*uo*vo;
			P1z[i][j] = Pz[i][j]*(1.0-uo)*(1.0-vo) + Pz[i+1][j]*uo*(1-vo) + Pz[i][j+1]*vo*(1-uo)+Pz[i+1][j+1]*uo*vo;

			if(k==2)
			{
				Pux[i][j] = Px[i][j]*(-1)*(1-vo) + Px[i+1][j]*(1)*(1-vo) + Px[i][j+1]*vo*(-1)+Px[i+1][j+1]*(1)*vo;
				Puy[i][j] = Py[i][j]*(-1)*(1-vo) + Py[i+1][j]*(1)*(1-vo) + Py[i][j+1]*vo*(-1)+Py[i+1][j+1]*(1)*vo;
				Puz[i][j] = Pz[i][j]*(-1)*(1-vo) + Pz[i+1][j]*(1)*(1-vo) + Pz[i][j+1]*vo*(-1)+Pz[i+1][j+1]*(1)*vo;

				Pvx[i][j] = Px[i][j]*(1-uo)*(-1) + Px[i+1][j]*uo*(-1) + Px[i][j+1]*1*(1-uo)+Px[i+1][j+1]*uo*1;
				Pvy[i][j] = Py[i][j]*(1-uo)*(-1) + Py[i+1][j]*uo*(-1) + Py[i][j+1]*1*(1-uo)+Py[i+1][j+1]*uo*1;
				Pvz[i][j] = Pz[i][j]*(1-uo)*(-1) + Pz[i+1][j]*uo*(-1) + Pz[i][j+1]*1*(1-uo)+Pz[i+1][j+1]*uo*1;
			}
		}
		__syncthreads();	
		if(i<=2-k&&j<=2-k)	Px[i][j] = P1x[i][j];
		if(i<=2-k&&j<=2-k)	Py[i][j] = P1y[i][j];
		if(i<=2-k&&j<=2-k)	Pz[i][j] = P1z[i][j];
		__syncthreads();	
	}
	
// Point and first order derivates
	if(i==0&&j==0)
	{
		float Sx = 	    Px[0][0];
		float Sy = 	    Py[0][0];
		float Sz = 	    Pz[0][0];

		float Sux = 3.0*Pux[0][0];
		float Suy = 3.0*Puy[0][0];
		float Suz = 3.0*Puz[0][0];

		float Svx = 3.0*Pvx[0][0];
		float Svy = 3.0*Pvy[0][0];
		float Svz = 3.0*Pvz[0][0];
	
		opx[jb*100 + ib] = Sx;
		opy[jb*100 + ib] = Sy;
		opz[jb*100 + ib] = Sz;
		
		ofx[jb*100 + ib]  = Suy*Svz-Suz*Svy;//xNormal	
		ofy[jb*100 + ib]  = Suz*Svx-Sux*Svz;//yNormal	
		ofz[jb*100 + ib]  = Sux*Svy-Suy*Svx;//zNormal	
	}
}

void seqkernelFoo(Point CP[49],float u_knots[9],float v_knots[9],Point *op);

int main()
{
	//Chosing correct interval of knots and control points for the following value

	Point op[10000];// Point input array
	Point gridop[100][100],gridof[100][100];// Point and normal offset

	Point P[49] = {//Grid of control points
	Point(0,0,0), Point(10,0,0), Point(20,0,0), Point(30,0,0), Point(40,0,0),Point(50,0,0),Point(60,0,0),
	Point(0,10,0),Point(10,10,10),Point(20,10,30),Point(30,10,25),Point(40,10,15),Point(50,10,15),Point(60,10,5),
	Point(0,20,5),Point(10,20,20),Point(20,20,40),Point(30,20,45),Point(40,20,35),Point(50,20,30),Point(60,20,15),
	Point(0,30,15),Point(10,30,20),Point(20,30,35),Point(30,30,40),Point(40,30,45),Point(50,30,35),Point(60,30,25),
	Point(0,40,10),Point(10,40,30),Point(20,40,35),Point(30,40,35),Point(40,40,50),Point(50,40,40),Point(60,40,20),
	Point(0,50,5),Point(10,50,15),Point(20,50,15),Point(30,50,25),Point(40,50,30),Point(50,50,25),Point(60,50,15),
	Point(0,60,0),Point(10,60,5),Point(20,60,10),Point(30,60,15),Point(40,60,20),Point(50,60,15),Point(60,60,5),
	};

	float Px[49],Py[49],Pz[49];// Input control points
	float opx[10000],opy[10000],opz[10000];// Point on the fabric surface
	float ofx[10000],ofy[10000],ofz[10000];// Offset Point for yarn

	for(int i = 0;i<49;i++)
	{
		Px[i] = P[i].x;
		Py[i] = P[i].y;
		Pz[i] = P[i].z;
	}
	
	
	float u[9] = {0,0,0,1,2,3,4,4,4};
	float v[9] = {0,0,0,1,2,3,4,4,4};

//	int sizeP = 49 * sizeof(Point);
	int sizePx =  49 * sizeof(float);
//	int sizeop = 10000 * sizeof(Point);
	int sizeopx =  10000 * sizeof(float);
	int sizeofx =  10000 * sizeof(float);
	int sizek = 9 * sizeof(float);

//	Point *cuda_P; = cudaMalloc(&cuda_P,sizeP);
	float *cuda_Px;  cudaMalloc(&cuda_Px,sizePx);
	float *cuda_Py;  cudaMalloc(&cuda_Py,sizePx);
	float *cuda_Pz;  cudaMalloc(&cuda_Pz,sizePx);

//	Point *cuda_op; = (Point *) cudaMalloc(sizeop);
	float *cuda_opx;  cudaMalloc(&cuda_opx,sizeopx);
	float *cuda_opy;  cudaMalloc(&cuda_opy,sizeopx);
	float *cuda_opz;  cudaMalloc(&cuda_opz,sizeopx);

	float *cuda_ofx;  cudaMalloc(&cuda_ofx,sizeofx);
	float *cuda_ofy;  cudaMalloc(&cuda_ofy,sizeofx);
	float *cuda_ofz;  cudaMalloc(&cuda_ofz,sizeofx);

	float *cuda_u_knots; cudaMalloc(&cuda_u_knots,sizek);
	float *cuda_v_knots; cudaMalloc(&cuda_v_knots,sizek);

	dim3 DimGrid(100, 100); // 10000 thread blocks
	dim3 DimBlock(4, 4); // 16 threads per block

	//defining variables for timing
	cudaEvent_t startEvent_inc, stopEvent_inc;
	cudaEventCreate(&startEvent_inc);
	cudaEventCreate(&stopEvent_inc);
	float elapsedTime_inc,seqTime;

	cudaEventRecord(startEvent_inc, 0); // starting timing for inclusive

// 	cudaMemcpy(cuda_P, P, sizeP, cudaMemcpyHostToDevice);
  	cudaMemcpy(cuda_Px, Px, sizePx, cudaMemcpyHostToDevice);
  	cudaMemcpy(cuda_Py, Py, sizePx, cudaMemcpyHostToDevice);
  	cudaMemcpy(cuda_Pz, Pz, sizePx, cudaMemcpyHostToDevice);

  	cudaMemcpy(cuda_u_knots, u, sizek, cudaMemcpyHostToDevice);
  	cudaMemcpy(cuda_v_knots, v, sizek, cudaMemcpyHostToDevice);


//	kernelFoo<<<DimGrid,DimBlock>>>(cuda_P,cuda_u_knots,cuda_v_knots,cuda_op);
	kernelFoo<<<DimGrid,DimBlock>>>(cuda_Px,cuda_Py,cuda_Pz,cuda_u_knots,cuda_v_knots,cuda_opx,cuda_opy,cuda_opz,cuda_ofx,cuda_ofy,cuda_ofz);

//	cudaMemcpy(op, cuda_op,sizeop, cudaMemcpyDeviceToHost);
	cudaMemcpy(opx, cuda_opx,sizeopx, cudaMemcpyDeviceToHost);
	cudaMemcpy(opy, cuda_opy,sizeopx, cudaMemcpyDeviceToHost);
 	cudaMemcpy(opz, cuda_opz,sizeopx, cudaMemcpyDeviceToHost);

	cudaMemcpy(ofx, cuda_ofx,sizeofx, cudaMemcpyDeviceToHost);
	cudaMemcpy(ofy, cuda_ofy,sizeofx, cudaMemcpyDeviceToHost);
 	cudaMemcpy(ofz, cuda_ofz,sizeofx, cudaMemcpyDeviceToHost);

	cudaEventRecord(stopEvent_inc, 0);  //ending timing for inclusive
	cudaEventSynchronize(stopEvent_inc);
	cudaEventElapsedTime(&elapsedTime_inc, startEvent_inc, stopEvent_inc);


	stopwatch<std::milli, float> sw;
	// Start the timer
	sw.start();
	seqkernelFoo(P,u,v,op);
	sw.stop();
	seqTime = sw.count();

	printf("Cuda time (ms)= %f\n",elapsedTime_inc);
	printf("Sequential time (ms) = %f\n",seqTime);
	float tol = 1e-4;
	for(int i = 0;i<10000;i++)
	{	
		float tolx = fabs(op[i].x - opx[i]);
		float toly = fabs(op[i].y - opy[i]);
		float tolz = fabs(op[i].z - opz[i]);
		if(tolx>tol||toly>tol||tolz>tol)
		{	printf("Tolerence problem detected : ");
			printf("i = %i\t",i);
			printf("%f %f %f\t",op[i].x,op[i].y,op[i].z);
			printf("%f %f %f\n",opx[i],opy[i],opz[i]);
		}
	}
	for(int i = 0;i<100;i++)
	{
	for(int j = 0;j<100;j++)
	{
		gridop[i][j] = Point(opx[100*j+i],opy[100*j+i],opz[100*j+i]);
		gridof[i][j] = Point(ofx[100*j+i],ofy[100*j+i],ofz[100*j+i]);
	}
	}
	FILE* fp = fopen("op.data","w");
	for(int i = 0;i<100;i+=10)
	{
	for(int j = 0;j<100;j+=10)
	{
		fprintf(fp,"%f\t%f\t%f\n",gridop[i][j].x,gridop[i][j].y,gridop[i][j].z);
	} fprintf(fp,"\n");
	}
	fclose(fp);
	float d1;		
	fp = fopen("fabric.data","w");
	for(int i = 0;i<100;i++)// Y-direction Yarns
	{
	for(int j = 0;j<100;j++)
	{
		if(j%2==0) d1 = -1;//Alternating yarns
		else d1 =  1;
		if(i%2==0) d1 = -d1;
		Point yP = gridop[i][j] + d1*gridof[i][j]/norm(gridof[i][j]);//yarnPoint
		fprintf(fp,"%f\t%f\t%f\n",yP.x,yP.y,yP.z);
	} fprintf(fp,"\n\n");
	}fprintf(fp,"\n\n");
	for(int j = 0;j<100;j++)// X-direction Yarns
	{
	for(int i = 0;i<100;i++)
	{
		if(i%2==0) d1 = 1;//Alternating yarns
		else d1 =  -1;
		if(j%2==0) d1 = -d1;
		Point yP = gridop[i][j] + d1*gridof[i][j]/norm(gridof[i][j]);//yarnPoint
		fprintf(fp,"%f\t%f\t%f\n",yP.x,yP.y,yP.z);
	} fprintf(fp,"\n\n");
	}fprintf(fp,"\n\n");

	fclose(fp);

 	return 0;
}
void seqkernelFoo(Point CP[49],float u_knots[9],float v_knots[9],Point *op)
{
	float u[6],v[6],uf,vf;int iff,jff;
	Point P[4][4],P1[4][4],Pu[4][4],Pv[4][4];

	for(int ib = 0;ib<100;ib++)
	{
	for(int jb = 0;jb<100;jb++)
	{

		uf = ib*4.0/100.0 + 0.5*4.0/100.0; 
		vf = jb*4.0/100.0 + 0.5*4.0/100.0;

		for(int ii = 0;ii<8;ii++)   if((u_knots[ii]<=uf)&&(uf<u_knots[ii+1])) iff = ii;
		for(int jj = 0;jj<8;jj++)	if((v_knots[jj]<=vf)&&(vf<v_knots[jj+1])) jff = jj;

		for(int ii = 0;ii<6;ii++)   u[ii] = u_knots[ii + iff - 2];
		for(int jj = 0;jj<6;jj++)	v[jj] = v_knots[jj + jff - 2];
			
		for(int i = 0;i<4;i++)
		{
		for(int j = 0;j<4;j++)
		{
			int ii = i + iff - 2;
			int jj = j + jff - 2;
			P[i][j] = CP[7*jj+ii];
		}
		}
		for(int k = 0;	k<=2;	k++)
		{
			for(int i = 0;i<=2-k;i++)
			{
			for(int j = 0;j<=2-k;j++)
			{
	
				float uo = (uf-u[i+k])/(u[i+3]-u[i+k]);
				float vo = (vf-v[j+k])/(v[j+3]-v[j+k]);
				
				P1[i][j] = P[i][j]*(1.0-uo)*(1.0-vo) + P[i+1][j]*uo*(1-vo) + P[i][j+1]*vo*(1-uo)+P[i+1][j+1]*uo*vo;
	
				if(k==2)
				{
					Pu[i][j] = P[i][j]*(-1)*(1-vo) + P[i+1][j]*(1)*(1-vo) + P[i][j+1]*vo*(-1)+P[i+1][j+1]*(1)*vo;
					Pv[i][j] = P[i][j]*(1-uo)*(-1) + P[i+1][j]*uo*(-1) + P[i][j+1]*1*(1-uo)+P[i+1][j+1]*uo*1;
				}
			}
			}
			
			for(int i = 0;i<=2-k;i++){
			for(int j = 0;j<=2-k;j++){
				P[i][j] = P1[i][j];}}
		}
		// Point and first order derivates
		Point S = 	    P[0][0];
		Point Su = 3.0*Pu[0][0];
		Point Sv = 3.0*Pv[0][0];
	
		op[jb*100+ib] = S;

	}
	}
}


