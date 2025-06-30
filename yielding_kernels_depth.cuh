#ifdef DOUBLE_PRECISION
typedef double REAL;
typedef double2 REAL2;
typedef double3 REAL3;
typedef cufftDoubleComplex COMPLEX;
#else
typedef float REAL;
typedef float2 REAL2;
typedef float3 REAL3;
typedef cufftComplex COMPLEX;
#endif

__device__ REAL MU=(FF*0.5);

/*--------GLOBAL VARIABLES--------*/
__device__ unsigned int philox_seed; //133227701;

/*--------RNG ROUTINES--------*/
__device__
REAL DeviceBoxMuller(const REAL u1, const REAL u2)
{
	REAL r = sqrt( -2.0*log(u1) );
	REAL theta = 2.0*M_PI*u2;
	return r*sin(theta);
}

__device__
void PhiloxRandomQuartet(const unsigned long long int index, const unsigned int time, float *r1, float *r2, float *r3, float *r4)
{
	RNG4 rng;
	RNG4::ctr_type c_pair={{}};
	RNG4::key_type k_pair={{}};
	RNG4::ctr_type r_quartet;

		// keys = threadid
		k_pair[0]= index;
		// time counter
		c_pair[0]= time;
		c_pair[1]= philox_seed; // eventually do: PHILOX_SEED + sample;
		// random number generation
		r_quartet = rng(c_pair, k_pair);

		*r1= u01_open_open_32_53(r_quartet[0]);
		*r2= u01_open_open_32_53(r_quartet[1]);
		*r3= u01_open_open_32_53(r_quartet[2]);
		*r4= u01_open_open_32_53(r_quartet[3]);
}

__device__
void PhiloxRandomPair(const unsigned long long int index, const unsigned int time, float *r1, float *r2)
{
	RNG2 rng;
	RNG2::ctr_type c_pair={{}};
	RNG2::key_type k_pair={{}};
	RNG2::ctr_type r_pair;

		// keys = threadid
		k_pair[0]= index;
		// time counter
		c_pair[0]= time;
		c_pair[1]= philox_seed; // eventually do: PHILOX_SEED + sample;
		// random number generation
		r_pair = rng(c_pair, k_pair);

		*r1= u01_open_open_32_53(r_pair[0]);
		*r2= u01_open_open_32_53(r_pair[1]);
}


//---------- DEVICE FUNCTIONS -----------//

__device__ float Random_Width(const float r1){
	float r = EE*(1+r1);
	return r;
}


//////////-------------------------Update Kernels------------------///////////////////////////

__global__ void Kernel_InitForcePoints(REAL3* d_pinn_force_points, unsigned int strain_idex){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int idy = blockIdx.y*blockDim.y+threadIdx.y;
	if ( idx < LX && idy < LY){
		int index = idy*LX + idx;
		float r1, r2;
		PhiloxRandomPair(index, strain_idex, &r1, &r2);
		d_pinn_force_points[index].x =-EE*0.5*(1+r1); //-EE*INITWELL*(1+r1); ///*0.1*(1+r1); for deep initial state
		d_pinn_force_points[index].y = EE*0.5*(1+r1); //EE*INITWELL*(1+r1);
		d_pinn_force_points[index].z = -log(r2);
	}
}


__global__ void Kernel_RelaxToBottom(const REAL3* d_pinn_force_points, REAL* d_strain){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int idy = blockIdx.y*blockDim.y+threadIdx.y;
	if ( idx < LX && idy < LY){
		int index = idy*LX + idx;

		REAL el = d_pinn_force_points[index].x;
		REAL er = d_pinn_force_points[index].y;
		d_strain[index]=(el+er)/2;
	}
}		


__global__ void Kernel_EvaluatePinningForces_mono(unsigned int* d_strain_index, unsigned int* d_strain_jumps, const REAL* d_strain, REAL3* d_pinn_force_points, REAL* d_pinn_force, REAL* d_well_center){

	// compute idx and idy, the location of the element in the original dLX*dLY array 
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int idy = blockIdx.y*blockDim.y+threadIdx.y;
	if ( idx < LX && idy < LY){
		int index = idy*LX + idx;
		unsigned int pos; //well_key;

		float r1, r2;

		REAL el = d_pinn_force_points[index].x;
		REAL er = d_pinn_force_points[index].y;
		REAL e = d_strain[index];

		if(e>er){
			el=er;

			pos = d_strain_index[index];
			PhiloxRandomPair(index, pos+1, &r1, &r2);
			
			if(pos%2==0){
				er = er + 1.0; //Random_Width(r1);
				d_pinn_force_points[index].z = 1; //1 + ETA0*(0.5-r2); //1; //-log(r2) + ETA0; //
			}else{
				er = er + (1 -log(r2));
				d_pinn_force_points[index].z = 0;
			}

			d_pinn_force_points[index].x=el;
			d_pinn_force_points[index].y=er;

			d_strain_index[index]++;
			d_strain_jumps[index]++;
			
		//	printf("left %f, strain %f, right %f, ran %f\n", el, e, er, r1);
			
		}

		if(e<el){
			er=el;

			pos = d_strain_index[index];
			PhiloxRandomPair(index, pos-1, &r1, &r2);

			if(pos%2==0){
				el = el - 1.0; //Random_Width(r1);
				d_pinn_force_points[index].z = 1; // + ETA0*(0.5-r2); //1; //-log(r2) + ETA0; //
			}else{
				el = el - (1 - log(r2));
				d_pinn_force_points[index].z = 0;
			}
			
			d_pinn_force_points[index].x=el;
			d_pinn_force_points[index].y=er;
			
			d_strain_index[index]--; // es los correcto a usar!!
			//d_strain_index[index]++; // solo para prueba de no-memoria
			d_strain_jumps[index]++;
		}


		#ifdef SINUSOIDAL
		d_pinn_force[index] = d_pinn_force_points[index].z*FF*sin(M_2PI*(e-el)/(er-el)); //FF*((er-el)/EE)*sin(M_2PI*(e-el)/(er-el));
		#endif
		
		#ifdef PARABOLAS
		d_pinn_force[index] = d_pinn_force_points[index].z*FF*((el+er)/2.0 - e) ; //FF*((el+er)/2.0 - e) ;
		#endif
	}
}


__global__ void Kernel_EvaluatePinningForces_opt(const unsigned int* d_strain_index, const REAL* d_strain, REAL3* d_pinn_force_points, REAL* d_pinn_force)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int idy = blockIdx.y*blockDim.y+threadIdx.y;
	if ( idx < LX/2 && idy < LY){
		int index = idy*(LX/2) + idx;
		unsigned int pos = d_strain_index[index];

		float r1, r2;
		PhiloxRandomPair(index, pos, &r1, &r2);

		for(unsigned int i=0; i<2; i++){

			float ran = (i==0)*r1+(i==1)*r2;

			REAL el = d_pinn_force_points[index+i*NN/2].x;
			REAL er = d_pinn_force_points[index+i*NN/2].y;
			REAL e = d_strain[index+i*NN/2];

			//BRANCHING-FREE version
			REAL el_new = (e>er)*er + (e<el)*(el-EE*(1+ran)) + ((!(e>er))&&(!(e<el)))*el;
			REAL er_new = (e>er)*(er+EE*(1+ran)) + (e<el)*el + ((!(e>er))&&(!(e<el)))*er;
			d_pinn_force_points[index+i*NN/2].x=el_new;
			d_pinn_force_points[index+i*NN/2].y=er_new;
			d_pinn_force_points[index+i*NN/2].z = -log(r2);

			#ifdef SINUSOIDAL
			d_pinn_force[index+i*NN/2] = d_pinn_force_points[index+i*NN/2].z*FF*((er_new-el_new)/EE)*sin(M_2PI*(e-el_new)/(er_new-el_new));
			#endif

			#ifdef PARABOLAS
			d_pinn_force[index+i*NN/2] = d_pinn_force_points[index+i*NN/2].z*FF*((el_new+er_new)/2.0 - e)/EE;
			#endif
		}
	}
}


// Pointwise multiplication
static __global__ void Kernel_PointwiseMul(const REAL *a, const bool *b, REAL* c)
{
	// compute idx and idy, the location of the element in the original LX*LY array 
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int idy = blockIdx.y*blockDim.y+threadIdx.y;
	if ( idx < LX && idy < LY){
		int index = idy*LX + idx;
		c[index] = a[index] * b[index];
	}
}

// COMPLEX pointwise multiplication
static __global__ void Kernel_ComplexPointwiseMulAndScale(const COMPLEX* a, const REAL* b, COMPLEX* c, REAL scale)
{
	// compute idx and idy, the location of the element in the original LX*LY array 
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int idy = blockIdx.y*blockDim.y+threadIdx.y;
	if ( idx < LX && idy < (LY/2+1)){
		int index = idy*LX + idx;
		c[index].x = scale*(a[index].x * b[index]);
		c[index].y = scale*(a[index].y * b[index]);
	}
} 


static __global__ void Kernel_ComputeElasticForcesInFourierSpace_Eshelby(const COMPLEX* d_strain_c, COMPLEX* d_elast_force_c)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int j = blockIdx.y*blockDim.y+threadIdx.y;
	if ( i < LX && j < (LY/2+1)){
		REAL qx2 = 2-2*cos(M_2PI*i/(float)LX);
		REAL qy2 = 2-2*cos(M_2PI*j/(float)LY);

		int index = i*(LY/2+1)+j; 
		d_elast_force_c[index].x = ((i!=0)||(j!=0))*(-1*MU*(qx2-qy2)*(qx2-qy2)/((qx2+qy2)*(qx2+qy2)))*d_strain_c[index].x; 
		d_elast_force_c[index].y = ((i!=0)||(j!=0))*(-1*MU*(qx2-qy2)*(qx2-qy2)/((qx2+qy2)*(qx2+qy2)))*d_strain_c[index].y;
	}
}


__global__ void Kernel_ComputeElasticForcesInRealSpace_Laplacian(const REAL* d_strain, REAL* d_elast_force)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int j = blockIdx.y*blockDim.y+threadIdx.y;
	if ( i < LX && j < LY){
		int index = j*(LX) + i;

	REAL laplacian = d_strain[((i-1+LY)%LY)*LY + j] + d_strain[((i+1)%LY)*LY + j] + \
			d_strain[i*LY + (j-1+LX)%LX]+d_strain[i*LY + (j+1)%LX]-4*d_strain[index];

	d_elast_force[index]= MU*laplacian;

	}
}



static __global__ void Kernel_EulerIntegrationStepInFourier(const COMPLEX *d_dstrain_c, COMPLEX *d_strain_c, const REAL dt)
{
	// compute i and j, the location of the element in the original LX*LY array 
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int j = blockIdx.y*blockDim.y+threadIdx.y;
	if ( i < LX && j < (LY/2+1)){
		int index = j*LX + i;

		d_strain_c[index].x = d_strain_c[index].x + dt*d_dstrain_c[index].x;
		d_strain_c[index].y = d_strain_c[index].y + dt*d_dstrain_c[index].y;

	}
}


static __global__ void Kernel_EulerIntegrationStep(const REAL *d_elast_force, const REAL *d_pinn_force, REAL *d_strain, REAL applied_stress, const REAL dt)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int j = blockIdx.y*blockDim.y+threadIdx.y;
	if ( i < LX && j < LY){
		int index = j*LX + i;

//		d_strain[index] = d_strain[index] + dt*(d_elast_force[index] + applied_stress);
		d_strain[index] = d_strain[index] + dt*(d_elast_force[index] + d_pinn_force[index] + applied_stress);//

	}
}

static __global__ void Kernel_LocalForces_abs(const REAL *d_elast_force, const REAL *d_pinn_force, REAL *d_local_forces, REAL applied_stress)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int j = blockIdx.y*blockDim.y+threadIdx.y;
	if ( i < LX && j < LY){
		int index = j*LX + i;
			d_local_forces[index] = abs(d_elast_force[index] + d_pinn_force[index] + applied_stress);

	}
}


