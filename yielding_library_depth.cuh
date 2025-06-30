/*
 * yielding_library.cuh
 *
 *  started on: Oct 18, 2016
 *      Author: eze
 */

#ifndef YIELDING_LIBRARY_CUH
#define YIELDING_LIBRARY_CUH

// RNG: PHILOX
#include "RNGcommon/Random123/philox.h"
#include "RNGcommon/Random123/u01.h"

#include <iostream> 	/* std::cout, std::fixed */
#include <iomanip>	/* std::setprecision */
#include <cmath>
#include <fstream>
#include <string>

#include <assert.h>
#include <stdint.h> 	/* uint8_t */
#include <cufft.h>
#include <cstdlib>
#include <time.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/find.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/unique.h>
#include <thrust/fill.h>

#include "cuda_util.h"

//Define Philox Seed
typedef r123::Philox2x32 RNG2;
typedef r123::Philox4x32 RNG4;

using namespace std;
using namespace thrust::placeholders;

extern void init_genrand(unsigned long s);
extern double genrand_real3(void);
extern double gasdev_real3(void);

#include "yielding_kernels_depth.cuh"
//#include "yielding_kernels.cuh"

struct square : public thrust::unary_function<REAL,REAL>
{
  __host__ __device__
  REAL operator()(REAL x) const
  {
    return x * x;
  }
};

//////////////////////////////////////////////////////////////////////
class yielding_model
{
	private:
	cufftHandle plan_c2r;
	cufftHandle plan_r2c;

	public:
	// Pointer to host arrays (realspace)
	REAL *h_strain;
	REAL3 *h_force_points;
	REAL *h_force;
	unsigned int *h_counter;

	// Pointer to device arrays (realspace and fourierspace)
	unsigned int *d_strain_index;
	unsigned int *d_strain_jumps;
	REAL *d_strain;
	COMPLEX *d_strain_c;
	REAL *d_strain_aux;
	COMPLEX *d_strain_aux_c;
	REAL *d_dstrain;
	COMPLEX *d_dstrain_c;
	REAL3 *d_pinn_force_points;
	REAL *d_pinn_force;
	REAL *d_well_center;
	COMPLEX *d_pinn_force_c;
	REAL *d_elast_force;
	REAL *d_local_forces;
	COMPLEX *d_elast_force_c;

	// intializing the class
	yielding_model(){

		//Array sizes
//		size_t size_b  = NN * sizeof(bool);
		size_t size_ui = NN * sizeof(unsigned int);
		size_t size_r  = NN * sizeof(REAL);
		size_t size_c  = LX*(LY/2+1) * sizeof(COMPLEX);
		size_t size_rc = LX*(LY/2+1) * sizeof(REAL);

		// Allocate arrays on host
		h_strain = (REAL *)malloc(size_r);
		h_force_points = (REAL3 *)malloc(NN*sizeof(REAL3));  //el, er
		h_force = (REAL *)malloc(size_r);
		h_counter = (unsigned int *)malloc(size_ui);

		// Allocate arrays on device
		cudaMalloc((void **) &d_strain_index, size_ui);
		cudaMalloc((void **) &d_strain_jumps, size_ui);
		cudaMalloc((void **) &d_strain, size_r);
		cudaMalloc((void **) &d_strain_aux, size_r);		
		cudaMalloc((void **) &d_strain_c, size_c);
		cudaMalloc((void **) &d_strain_aux_c, size_c);				
		cudaMalloc((void **) &d_dstrain, size_r);
		cudaMalloc((void **) &d_dstrain_c, size_c);
		cudaMalloc((void **) &d_pinn_force_points, NN*sizeof(REAL3));
		cudaMalloc((void **) &d_pinn_force, size_r);
		cudaMalloc((void **) &d_well_center, size_r);
		cudaMalloc((void **) &d_elast_force, size_r);
		cudaMalloc((void **) &d_local_forces, size_r);
		cudaMalloc((void **) &d_pinn_force_c, size_c);
		cudaMalloc((void **) &d_elast_force_c, size_c);

		//Preventive fill everything with 0s
		for (unsigned int k=0; k<NN; k++){h_strain[k]=0.0; h_force[k]=0.0; h_counter[k]=0;}
		for (unsigned int k=0; k<NN; k++){h_force_points[k].x=0.0; h_force_points[k].y=0.0; h_force_points[k].z=0.0;}
		cudaMemset(d_strain_index, 0, size_ui);
		cudaMemset(d_strain_jumps, 0, size_ui);
		cudaMemset(d_strain, 0, size_r);
		cudaMemset(d_strain_aux, 0, size_r);				
		cudaMemset(d_strain_c, 0, size_c);
		cudaMemset(d_strain_aux_c, 0, size_c);				
		cudaMemset(d_dstrain, 0, size_r);
		cudaMemset(d_dstrain_c, 0, size_c);
		cudaMemset(d_pinn_force_points, 0, NN*sizeof(REAL3));
		cudaMemset(d_pinn_force, 0, size_r);
		cudaMemset(d_well_center, 0, size_r);		
		cudaMemset(d_elast_force, 0, size_r);
		cudaMemset(d_local_forces, 0, size_r);
		cudaMemset(d_pinn_force_c, 0, size_rc);
		cudaMemset(d_elast_force_c, 0, size_rc);

		// cuFFT plan
		#ifdef DOUBLE_PRECISION
		cufftPlan2d(&plan_c2r, LX, LY, CUFFT_Z2D);
		cufftPlan2d(&plan_r2c, LX, LY, CUFFT_D2Z);
		#else
		cufftPlan2d(&plan_c2r, LX, LY, CUFFT_C2R);
		cufftPlan2d(&plan_r2c, LX, LY, CUFFT_R2C);
		#endif
		//cufftSetCompatibilityMode(plan_c2r, CUFFT_COMPATIBILITY_NATIVE);
		//cufftSetCompatibilityMode(plan_r2c, CUFFT_COMPATIBILITY_NATIVE);
	}

////////////////////////////////////////////////////////////////////////////////////////////////

//////////-----------------------Initialize Functions----------------///////////////////////////

	//Initializes on Host and CpyToDevice
	//void Initialize(bool b, ifstream &stored_strains, ifstream &stored_pinn_force_points){
	void Initialize(){	
			
		for(int j = 0; j<LY; j++){
			for(int i = 0; i<LX; i++)
			{
				int ij = j*LX+i;
				h_force_points[ij].x = 0; //-EE*(a); //left //-FF*(1/8.0)*(a-0.5)+1; //
				h_force_points[ij].y = 0; //a //EE*(a); //right //FF*(1/8.0)*(a-0.5)+1; //
				h_force_points[ij].z = 0; //-log(b); //right //FF*(1/8.0)*(a-0.5)+1; //
				h_strain[ij] = 0; //0.1*a; // *(1+b/2); //STRESS_THRESHOLD; 


				for(int s=0; s<100; s++){
				float a = genrand_real3();
				float b = genrand_real3();
				
				h_force_points[ij].x = h_force_points[ij].y; 
				h_force_points[ij].y += (s%2==0)*1+(s%2==1)*(1-log(a)); //(b<0.5)*1+(b>=0.5)*(1-log(a)); 
				h_force_points[ij].z = (s%2==0)*1+(s%2==1)*0; //(b<0.5)*1+(b>=0.5)*0; 
				}
					
				h_strain[ij] = (h_force_points[ij].y + h_force_points[ij].x)/2.0; //0.1*a; // *(1+b/2); //STRESS_THRESHOLD; 

			}
		}
			
		REAL av_strain =0;			
		for(int j = 0; j<LY; j++){
			for(int i = 0; i<LX; i++)
			{
				int ij = j*LX+i;
				av_strain += h_strain[ij];			
			}
		}			
			
		for(int j = 0; j<LY; j++){
			for(int i = 0; i<LX; i++)
			{
				int ij = j*LX+i;
				h_strain[ij] = av_strain/(REAL) NN;
			}
		}			
		
		cudaMemcpy(d_strain, h_strain, NN*sizeof(REAL) ,cudaMemcpyHostToDevice);
		cudaMemcpy(d_pinn_force_points, h_force_points, NN*sizeof(REAL3) ,cudaMemcpyHostToDevice);
		
		thrust::device_ptr<unsigned int> strain_index_ptr (d_strain_index);
		thrust::fill(strain_index_ptr, strain_index_ptr+NN, 100000);
	}

	void InitializeOnDevice(){
		// Initialize Strain
		thrust::device_ptr<REAL> strain_ptr (d_strain);
		thrust::fill(strain_ptr, strain_ptr+NN, 0.0);
	}

	void InitFromFile(ifstream &stored_strains,ifstream &stored_pinn_force_points)
	{

		cudaMemcpy(d_strain, h_strain, NN*sizeof(REAL) ,cudaMemcpyHostToDevice);
		cudaMemcpy(d_pinn_force_points, h_force_points, NN*sizeof(REAL2) ,cudaMemcpyHostToDevice);
	}
	


////////////////////////////////////////////////////////////////////////////////////////////////

//////////---------------------------Print Functions-----------------///////////////////////////


	void PrintLocalStrain(ofstream &fstr){
		cudaMemcpy(h_strain, d_strain, LX*sizeof(REAL), cudaMemcpyDeviceToHost);
		for(int i=0;i<LX;i++){
			fstr << std::setprecision(16) << h_strain[i] << '\n';
		}
		fstr.close();
	}


	void PrintStrainJumpsCount(ofstream &fstr){
		cudaMemcpy(h_counter, d_strain_jumps, NN*sizeof(unsigned int), cudaMemcpyDeviceToHost);
		for(int i=0;i<NN;i++){
			fstr << h_counter[i] << '\n';
		}
		fstr.close();
	}

	void PrintStrain(ofstream &fstr){
		for(int i=0;i<NN;i++){
			fstr << std::setprecision(16) << h_strain[i] << '\n';
		}
		fstr.close();
	}

	void PrintForcePoints(ofstream &fstr){
		for(int i=0;i<NN;i++){
			fstr << std::setprecision(16) << h_force_points[i].x << " " << h_force_points[i].y << '\n';
		}
		fstr.close();
	}

	void PrintForce(ofstream &fstr){
		for(int i=0;i<NN;i++){
			fstr << std::setprecision(16) << h_force[i] << '\n';
		}
		fstr.close();
	}
	
////////////////////////////////////////////////////////////////////////////////////////////////

//////////-------------------------Update Functions------------------///////////////////////////

	void RelaxToBottom(){
	
		dim3 dimBlock(TILE_X, TILE_Y);
		dim3 dimGrid(LX/2/TILE_X, LY/TILE_Y);
		assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);
	
		Kernel_RelaxToBottom<<<dimGrid, dimBlock>>>(d_pinn_force_points, d_strain);
	
	}

	void EvaluatePinningForces(){
#ifdef OPT
		dim3 dimBlock(TILE_X, TILE_Y);
		dim3 dimGrid(LX/2/TILE_X, LY/TILE_Y);
		assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);
		Kernel_EvaluatePinningForces_opt<<<dimGrid, dimBlock>>>(d_strain_index, d_strain, d_pinn_force_points, d_pinn_force);
#else
		dim3 dimBlock(TILE_X, TILE_Y);
		dim3 dimGrid(LX/TILE_X, LY/TILE_Y);
		assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);
		Kernel_EvaluatePinningForces_mono<<<dimGrid, dimBlock>>>(d_strain_index, d_strain_jumps, d_strain, d_pinn_force_points, d_pinn_force, d_well_center);
#endif
	}
	
	void ComputeElasticForces(){
	
	
		dim3 dimBlock(TILE_X, TILE_Y);
		dim3 dimGrid(LX/TILE_X, LY/TILE_Y/2+1);
		assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);

		Kernel_ComputeElasticForcesInFourierSpace_Eshelby<<<dimGrid, dimBlock>>>(d_strain_c, d_elast_force_c);

	}


	void EulerIntegrationStep(REAL applied_stress){
		dim3 dimBlock(TILE_X, TILE_Y);
		dim3 dimGrid(LX/TILE_X, LY/TILE_Y);
		assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);

		Kernel_EulerIntegrationStep<<<dimGrid, dimBlock>>>(d_elast_force, d_pinn_force, d_strain, applied_stress, TIME_STEP);
	}

	void TransformToFourierSpace(){
		#ifdef DOUBLE_PRECISION
		cufftExecD2Z(plan_r2c, d_strain, d_strain_c);
		#else
		cufftExecR2C(plan_r2c, d_strain, d_strain_c);
		#endif
	}

	void AntitransformFromFourierSpace(){
		#ifdef DOUBLE_PRECISION
		cufftExecZ2D(plan_c2r, d_elast_force_c, d_elast_force);
		#else
		cufftExecC2R(plan_c2r, d_elast_force_c, d_elast_force);		
		#endif
		//SCALE AFTER ANTITRANSFORMING
		thrust::device_ptr<REAL> d_elast_force_ptr (d_elast_force);
		thrust::transform(d_elast_force_ptr, d_elast_force_ptr+NN, d_elast_force_ptr, (1.0/REAL(NN))*_1);

	}

////////////////////////////////////////////////////////////////////////////////////////////////

//////////-------------------------Calculate Functions---------------///////////////////////////

	REAL ComputeStress(REAL gamma_p){

		thrust::device_ptr<REAL> force_ptr (d_pinn_force);
		REAL meanforce = thrust::reduce(force_ptr,force_ptr+NN)/(REAL)NN;
		return gamma_p - meanforce;

	}

	REAL ComputeStrainRate(REAL ext_stress){

		thrust::device_ptr<REAL> force_ptr (d_pinn_force);
		REAL meanforce = thrust::reduce(force_ptr,force_ptr+NN)/(REAL)NN;
		
		
		return ext_stress + meanforce; 
		
	}

	bool CheckRestCondition(REAL applied_stress){

		dim3 dimBlock(TILE_X, TILE_Y);
		dim3 dimGrid(LX/TILE_X, LY/TILE_Y);
		assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);

		Kernel_LocalForces_abs<<<dimGrid,dimBlock>>>(d_elast_force, d_pinn_force, d_local_forces, applied_stress);
		thrust::device_ptr<REAL> absolute_ptr(d_local_forces);

		bool is_someone_unstable = thrust::any_of(absolute_ptr,absolute_ptr+NN, _1 > STOP_CRITERION); 
		return (is_someone_unstable);
	}

	
	REAL ComputeMeanStrain(){

		thrust::device_ptr<REAL> strain_ptr (d_strain);
		REAL mean_strain = thrust::reduce(strain_ptr,strain_ptr+NN)/(REAL)NN;
		
		return mean_strain;
	}


	REAL ComputeMeanWidth(){	

		thrust::device_ptr<REAL> strain_ptr (d_strain);
		REAL mean_strain = thrust::reduce(strain_ptr,strain_ptr+NN)/(REAL)NN;

		REAL sum_of_squares = \
		  thrust::reduce(thrust::make_transform_iterator(strain_ptr, square()), \
		  thrust::make_transform_iterator(strain_ptr+NN, square()));

		REAL mean_width = sqrt(sum_of_squares/REAL(NN)-mean_strain*mean_strain);
	
		return mean_width;
	}	


	REAL ComputeStrainFluctuations(){

		thrust::device_ptr<REAL> dstrain_ptr (d_dstrain);
		REAL sum_of_squares = \
		  thrust::reduce(thrust::make_transform_iterator(dstrain_ptr, square()), \
		  thrust::make_transform_iterator(dstrain_ptr+NN, square()));

		return sum_of_squares;
	}
	

////////////////////////////////////////////////////////////////////////////////////////////////

//////////---------------------------Copy Functions------------------///////////////////////////

	/* transfer from GPU to CPU memory */
	void CpyDeviceToHost(){
		size_t size_r = NN * sizeof(REAL);
		size_t size_r3 = NN * sizeof(REAL3);
		cudaMemcpy(h_strain, d_strain, size_r, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_force_points, d_pinn_force_points, size_r3, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_force, d_pinn_force, size_r, cudaMemcpyDeviceToHost);
	}

////////////////////////////////////////////////////////////////////////////////////////////////

//////////---------------------------Cleaning Issues-----------------///////////////////////////

	~yielding_model(){

		/* frees CPU memory */
		free(h_strain);
		free(h_force_points);
		free(h_force);

		/* frees GPU memory */
		cudaFree(d_strain_index);
		cudaFree(d_strain);
		cudaFree(d_strain_c);
		cudaFree(d_dstrain);
		cudaFree(d_dstrain_c);
		cudaFree(d_pinn_force_points);
		cudaFree(d_pinn_force);
		cudaFree(d_well_center);
		cudaFree(d_pinn_force_c);
		cudaFree(d_elast_force_c);

		cufftDestroy(plan_r2c);
		cufftDestroy(plan_c2r);

	};
};


#endif /*  YIELDING_LIBRARY_CUH */
