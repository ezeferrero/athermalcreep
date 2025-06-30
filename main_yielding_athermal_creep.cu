/*
 * Yielding Athermal Creep
 *
 * Started on: Jan, 2022 by ezeferrero
 * Copyright (C) 2025 Ezequiel E. Ferrero, Eduardo A. Jagla
*/

/*
 * main_yielding_athermal_creep.cu, together with its libraries yielding_library.cuh & yielding_kernels_depth.cuh,
 * is a CUDA implementation of a 2D elasto-plastic (EP) model.
 * The program simulates a 2-dimensional amorphous solid sheared at a fix apllied stress while
 * its elastic properties (in particular the elastic shear modulus) can be oscillated in time.
 * A scalar field represents the value of the shear-stress in each block of a square lattice.
 * An overdamped Eulerian dynamics is proposed for the scalar shear-stresses.
 * Long-range interactions are solved using a pseudo-spectral method, that renders the equation
 * of movement to be local in Fourier space. We make intensive use of the cuFFT library.
 * Stochasticity is introduced in a disorder potential that mimics the local yielding thresholds,
 * to that end the counter based PRNG "Philox" is used.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * This code was originally implemented for: 
 * "Cyclic Perturbations Facilitate Athermal Creep in Yield-Stress Materials",
 * Ezequiel E. Ferrero and Eduardo A. Jagla
 * https://arxiv.org/abs/2501.07782
 * Soft Matter ... (2025)
 * https://doi.org/10.1039/D5SM00186B
 *
 * Please cite when appropriate.
 */




#ifndef LX
#define LX 256
#endif
#ifndef LY
#define LY LX 
#endif

#define NN (LX*LY)
#define SCALE (1.0/NN)

#ifndef TIME_STEP
#define TIME_STEP 0.01
#endif

#ifndef FF
#define FF 1.0 //10.0
#endif
#define BB 1.0

#ifndef EE
#define EE 1.0 //0.1 //0.01
#endif

#ifndef KK
#define KK 1.0
#endif

#define ETA0 0.1

#ifndef KAPPA
#define KAPPA 10.0 //1.0 //
#endif

#ifndef INITWELL
#define INITWELL 1.0 //0.3
#endif

#ifndef STRAIN_TRANSIENT
#define STRAIN_TRANSIENT 0
#endif

#ifndef DATA_T
#define DATA_T 0.01 //0.05
#endif

#define STOP_CRITERION (6.4e-4) //(8e-3)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_2PI
#define M_2PI (2*3.14159265358979323846)
#endif


//Tunne this to optimize
#define TILE_X 64
#define TILE_Y 8

#ifndef CUDA_DEVICE
#define CUDA_DEVICE 0
#endif

// Functions
#define MAX(a,b) (((a)<(b))?(b):(a))	// maximum
#define MIN(a,b) (((a)<(b))?(a):(b))	// minimum
#define ABS(a) (((a)< 0)?(-(a)):(a))	// absolute value

// Hardware parameters for Tesla c2075 (GF100) cc 2.0
#define SHARED_PER_BLOCK 49152
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 1024
#define BLOCKS_PER_GRID 65535

//#include "yielding_library.cuh"
#include "yielding_library_depth.cuh"
#include "timers.hpp"

REAL kappa = KAPPA;
//REAL KK=1.0;
REAL kl = 1.0; //0.5; //2.0;
REAL ks = 0.0000000000000000001; //0001;
unsigned int nofoscilations = 100; //1000; //4000; //100; //600; //SHOULD BE EVEN
int period = 100;

bool logtimes(unsigned long long int t, unsigned int logincrement){
	bool a = ( (t % logincrement == 0) || ((t) % (logincrement/2) == 0 ) || \
			((t) % (logincrement/3) == 0) || ((t) % (logincrement/5) == 0 ) || \
			((t) % (logincrement/7) == 0 ) );
	return a;
}

#ifdef EQUILIBRATION
void Equilibrate(yielding_model &T, const REAL ext_stress, const unsigned long long int t_equilibrate){

	for(unsigned long long int i=0; i<t_equilibrate; i++){
		T.EvaluatePinningForces();
		T.TransformToFourierSpace();
		T.ComputeElasticForces();
		T.AntitransformFromFourierSpace();
#ifdef FIX_EXT_STRESS
		T.EulerIntegrationStep(ext_stress);
#else
		T.EulerIntegrationStep(ext_stress);
#endif
	}
	
	T.RelaxToBottom();
}
#endif

#ifdef FIX_EXT_STRESS

void OcillatoryPhase(yielding_model &T, const REAL ext_stress, const unsigned int nofoscilations, const REAL ks, const REAL kl){

//////// OUTPUT FILE
	char filename[128];
		sprintf(filename, "displacement-nofoscillations%6.6f_FF%3.3f_LX%dLY%d_dt%4.4f.dat", ext_stress, FF, LX , LY, TIME_STEP);
		ofstream outss(filename);
		outss << "## << h*nofoscilations+l << " " << mean_strain - saved_mean_strain << " " << mean_width << " " << variable_strain_rate << " " << ext_stress << " " << mean_strain << " " << pp" << endl;

	outss << std::setprecision (15);


/// OSCILLATRY LOOP

	REAL applied_stress = ext_stress;
	REAL mean_strain, saved_mean_strain, variable_strain_rate, mean_width;
	REAL hKK;	
	REAL pp=period;

	for(unsigned int h=0; h<1; h++){		

		for(unsigned int l=0; l<nofoscilations; l++){
		
			hKK=kl;
			
			for(unsigned int p=0; p<pp+1; p++){ 
			
				hKK = ks + (kl-ks)*(cos(2*M_PI*p/float(pp))+1)/2.0;
				cudaMemcpyToSymbol(MU, &hKK, sizeof(REAL), 0, cudaMemcpyHostToDevice);

				T.EvaluatePinningForces();
				T.TransformToFourierSpace();
				T.ComputeElasticForces();
				T.AntitransformFromFourierSpace();
				T.EulerIntegrationStep(applied_stress);
			}

			if(l==0){
				saved_mean_strain = T.ComputeMeanStrain();
			}
			 
			//Measurements
			if(l>0){		
				mean_strain = T.ComputeMeanStrain();	
				mean_width = T.ComputeMeanWidth();	
				variable_strain_rate=T.ComputeStrainRate(ext_stress);
				outss << h*nofoscilations+l << " " << mean_strain - saved_mean_strain << " " << mean_width << " " << variable_strain_rate << " " << ext_stress << " " << mean_strain << " " << pp << endl; //"\n";
				saved_mean_strain = mean_strain;
			}

		}
		pp = pp*5;		
		
	}
}


void TimeLoop(yielding_model &T, const REAL ext_stress, const unsigned long long int t_equilibrate, \
			const unsigned long long int t_transient, const unsigned long long int t_run, \
			const unsigned long long int t_data){

	char filename[128];

	sprintf(filename, "strainrate-time_stress%6.6f_FF%3.3f_LX%dLY%d_dt%4.4f.dat", ext_stress, FF, LX , LY, TIME_STEP);
	ofstream outss(filename);
	outss << "## << t*TIME_STEP << " " << variable_strain_rate << " " << ext_stress << " " << mean_width << " " << mean_strain " << endl;
	outss << std::setprecision (15);

	REAL variable_strain_rate;
	REAL applied_stress = ext_stress;

	for(unsigned long long int i=0; i<t_transient; i++){
		T.EvaluatePinningForces();
		T.TransformToFourierSpace();
		T.ComputeElasticForces();
		T.AntitransformFromFourierSpace();
		T.EulerIntegrationStep(applied_stress);
	}

	unsigned long long int t=0;
	unsigned int logincrement=1;

	REAL mean_strain, mean_width; 
	REAL saved_mean_strain=0.0; 

	while(t<t_run+1){

		//Measurements
		if(t%t_data==0){
			variable_strain_rate=T.ComputeStrainRate(ext_stress);
			mean_strain = T.ComputeMeanStrain();
			mean_width = T.ComputeMeanWidth();	
			outss << t*TIME_STEP << " " << variable_strain_rate << " " << ext_stress << " " << mean_width << " " << mean_strain - saved_mean_strain << endl; //"\n";
			saved_mean_strain = mean_strain;
		}

		//Dynamics
		T.EvaluatePinningForces();
		T.TransformToFourierSpace();
		T.ComputeElasticForces();
		T.AntitransformFromFourierSpace();
		T.EulerIntegrationStep(applied_stress);
		t++;

		if (t == logincrement*10){ logincrement*=10 ;} //outss.flush();
	}
	outss.close();
}



#else

unsigned long long int TimeLoop(yielding_model &T, const REAL gamma_p, const unsigned long long int t_equilibrate, \
			const unsigned long long int t_transient, const unsigned long long int t_run, \
			const unsigned long long int t_data, double &ssstress, double &ssstress2){

	char filename[128];


	sprintf(filename, "stress-strain_gdot%6.6f_FF%3.3f_LX%dLY%d_dt%4.4f.dat", gamma_p, FF, LX , LY, TIME_STEP);
	ofstream outss(filename);
	outss << "## << t*TIME_STEP*gamma_p << " " << sigmaaverage << " " << gamma_p " << endl;
	outss << std::setprecision (15);

	REAL driving_coord = 0.0;
	REAL applied_stress, mean_strain;

	for(unsigned long long int i=0; i<t_transient; i++){
		T.EvaluatePinningForces();
		T.TransformToFourierSpace();
		T.ComputeElasticForces();
		T.AntitransformFromFourierSpace();

		driving_coord += gamma_p*TIME_STEP;		
		mean_strain = T.ComputeMeanStrain();
		applied_stress = kappa*(driving_coord - mean_strain);

		T.EulerIntegrationStep(applied_stress);
	}

	unsigned long long int count=0;
	unsigned long long int t=0;

	while(t<t_run+1){

	//Measurements
	if(t%t_data==0){
		REAL stress = T.ComputeStress(gamma_p);
		outss << t*TIME_STEP*gamma_p << " " << stress << " " << gamma_p << endl;
		if(t>t_run/2){
			ssstress+= stress;
			ssstress2+=stress*stress;
			count++;

			#ifdef PRINTFORCES
			if (t%(t_data*1000)==0){
			sprintf(filename, "Forces_gdot%6.6f_time%d_FF%3.3f_LX%dLY%d_dt%4.4f.dat", gamma_p, t, FF, LX , LY, TIME_STEP);
			ofstream outforces(filename);
			T.PrintForce(outforces);
			outforces.close();
			}
			#endif

		}
	}

	//Dynamics
		T.EvaluatePinningForces();
		T.TransformToFourierSpace();
		T.ComputeElasticForces();
		T.AntitransformFromFourierSpace();
			
		driving_coord += gamma_p*TIME_STEP;		
		mean_strain = T.ComputeMeanStrain();
		applied_stress = kappa*(driving_coord - mean_strain);

		T.EulerIntegrationStep(applied_stress);
		t++;
	}

	ssstress = ssstress/(REAL)count;
	ssstress2 = ssstress2/(REAL)count;

	outss.close();

	return count;
}
#endif


int ReadForcesList(float *myNumbers){

	// Forces will be read form a file containing them in the first line separated by commas
	FILE *file = fopen("forcevalues.txt", "r");
	if (file == NULL) {
        	perror("Unable to open file");
        return 1;
        }
  	char line[4096];
	if (fgets(line, sizeof(line), file) == NULL) {
		perror("Error reading from file");
		fclose(file);
		return 1;
	}
	fclose(file);
	
	printf("## Forces values read: %s\n", line);
    	
    	// Count the number of entries in the line
	    int count = 1; // Initialize with 1 for the first value
	    for (int i = 0; line[i] != '\0'; i++) {
		if (line[i] == ',') {
		    count++;
		}
	    }
    	
    	cout << "# count #forces: " << count << '\n' ;
    	
	    // Tokenize the line to extract the number of entries and the values
	    //float myNumbers[128]; // Assuming a maximum of 1024 values
	    char *token = strtok(line, ",");    	
		if (count > 128) {
			printf("Error: Too many entries in the file.\n");
			return 1;
        	}
        	
        if (token == NULL || sscanf(token, "%f", &myNumbers[0]) != 1 || myNumbers[0] <= 0) {
     	   printf("Error: Invalid number of entries.\n");
     	   return 1;
    	}
 
     // Parse and store the float values in the array
	    int valuesRead = 1; // Initialize with 1 for the first value
	    for (int i = 0; i < count-1; i++) {
	        token = strtok(NULL, ",");
	        if (token == NULL || sscanf(token, "%f", &myNumbers[valuesRead]) != 1) {
	            printf("Error: Invalid value at position %d.\n", valuesRead + 1);
	            return 1;
	        }
	        valuesRead++;
	    }

	    if (valuesRead != count) {
	        printf("Error: Number of values read does not match the specified count.\n");
	        return 1;
	    }
 
 	//int npoints = count;
        
        printf("##The array values are: ");
	    for (int i = 0; i < count; i++) {
        	printf("%.4f ", myNumbers[i]);
	    }	
    		printf("\n");
    		
    	return count;	

}


// main routine
int main(int argc, char **argv) {

	/*------------INPUT----------------*/
	if(argc < 5) {
		cerr << "\n\n\t\tUsage: " << argv[0] << " GDOT_OR_SIGMAEXT STRAIN_EQUILIBRATE STRAIN_RUN STRAIN_DATA kL kS [SEED_CPU_RNG] [SEED_PHILOX] [stored_strains (name with quotes)] [stored_force_points (name with quotes)] \n\n";
		return 1;
	}

	// TODO: Free M, let it be multiples of TILE, not necessarily power of 2.
	const REAL GDOT_OR_SIGMAEXT = atof(argv[1]);
	const REAL STRAIN_EQUILIBRATE = atof(argv[2]);
	const REAL STRAIN_RUN = atof(argv[3]);
	const REAL STRAIN_DATA = atof(argv[4]);
	ks = atof(argv[5]);
	kl = atof(argv[6]);
	period = atoi(argv[7]);
	const REAL minstress = atof(argv[8]);
	const REAL stressstep = atof(argv[9]);	
	
	unsigned int seed_random = ((unsigned int) time(NULL));
	if(argc > 9 && atoi(argv[10])!=55) seed_random = atoi(argv[10]);
	// argv[6}=55 allows us to use i>6 arguments without declaring argv[6]
	unsigned int h_philox_seed = ((unsigned int) time(NULL)); //PHILOX_SEED;
	if(argc > 10 && atoi(argv[11])!=55) h_philox_seed = atoi(argv[11]);
	cudaMemcpyToSymbol(philox_seed, &h_philox_seed, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);

	const REAL criticalstress = atof(argv[12]);

	//TODO: complete asserts
	assert(TILE_X%2==0);
	assert(TILE_Y%2==0);
	assert(LX%TILE_X==0);
	assert(LY%TILE_Y==0);
	assert(LX/2>=TILE_X);
	assert(LY/2>=TILE_Y);

// Set the GPGPU computing device
//	#ifdef CUDA_DEVICE
//	if (CUDA_DEVICE!=55) cudaSetDevice(CUDA_DEVICE);
//	#endif

// Choosing less "shared memory", more cache.
	//CUDA_SAFE_CALL(cudaThreadSetCacheConfig(cudaFuncCachePreferL1)); 

	std::setprecision (15);

	//print simulation parameters
	cout << "### LX ................................: " << LX << '\n';
	cout << "### LY ................................: " << LY << '\n';
	cout << "### Time Step..........................: " << TIME_STEP << '\n';
#ifdef FIX_EXT_STRESS
	cout << "### External Stress....................: " << GDOT_OR_SIGMAEXT << '\n';
#else
	cout << "### Shear rate ........................: " << GDOT_OR_SIGMAEXT << '\n';
#endif
	cout << "### Strain EQUILIBRATE.................: " << STRAIN_EQUILIBRATE << '\n';
	cout << "### Strain TRANSIENT...................: " << STRAIN_TRANSIENT << '\n';
	cout << "### Strain RUN.........................: " << STRAIN_RUN << '\n';
	cout << "### Strain DATA........................: " << STRAIN_DATA << '\n';
	cout << "### Force amplitude....................: " << FF << '\n';
	//cout << "### Elastic constant...................: " << KK << '\n';
	cout << "### Strain distance among wells........: " << EE << '\n';
//	cout << "### Data aquiring......................: " << DATA_T << '\n';
	cout << "### small k (ks).......................: " << ks << '\n';
	cout << "### large k (kl).......................: " << kl << '\n';
	cout << "### oscillation period ................: " << period << '\n';
	cout << "### KAPPA of driving when fix veloc  ..: " <<  KAPPA << '\n';
	#ifdef DOUBLE_PRECISION
	cout << "### DOUBLE PRECISION...................: " << '\n';
	#else
	cout << "### SINGLE PRECISION...................: " << '\n';
	#endif
	#ifdef SINUSOIDAL
	cout << "### SINUSOIDAL FORCE...................: " << '\n';
	#endif
	#ifdef PARABOLAS
	cout << "### CONCATENATED PARABOLAS FORCE.......: " << '\n';
	#endif
	cout << "### SEED MT.... .......................: " << seed_random << '\n';
	cout << "### SEED Philox .......................: " << h_philox_seed << endl;
	
	float myNumbers[128]; // Assuming a maximum of 1024 values
	int npoints = ReadForcesList(myNumbers);
	

	// Initialize Mersenne
	init_genrand(seed_random);

	// Our class
	yielding_model T;

	// Turn on cronometer
	cpu_timer clock;
	clock.tic();

#ifndef FIX_EXT_STRESS
	char flowcurvefilename [256];
	sprintf(flowcurvefilename, "flowcurve_gdot%8.8f_LX%dLY%d_dt%4.4f.dat", GDOT_OR_SIGMAEXT, LX , LY, TIME_STEP);
	ofstream outflow(flowcurvefilename);
	outflow << std::setprecision (15);
#endif

#ifdef FIX_EXT_STRESS
	unsigned long long int t_equilibrate=(int)((STRAIN_EQUILIBRATE)/TIME_STEP+0.5);
	unsigned long long int t_transient=(int)((STRAIN_TRANSIENT)/TIME_STEP+0.5);
	unsigned long long int t_run=(int)((STRAIN_RUN)/TIME_STEP+0.5);
	unsigned long long int t_data= (int)((STRAIN_DATA)/TIME_STEP+0.5);
	cout <<  "###" << t_equilibrate << " " <<  t_transient << " " << t_run << " " << t_data << " " << h_philox_seed << endl;
#else
	unsigned long long int t_equilibrate=(int)((STRAIN_EQUILIBRATE/GDOT_OR_SIGMAEXT)/TIME_STEP+0.5);
	unsigned long long int t_transient=(int)((STRAIN_TRANSIENT/GDOT_OR_SIGMAEXT)/TIME_STEP+0.5);
	unsigned long long int t_run=(int)((STRAIN_RUN/GDOT_OR_SIGMAEXT)/TIME_STEP+0.5);
	unsigned long long int t_data= (int)((STRAIN_DATA/GDOT_OR_SIGMAEXT)/TIME_STEP+0.5);
	cout <<  "###" << t_equilibrate << " " <<  t_transient << " " << t_run << " " << t_data << " " << h_philox_seed << endl;
#endif

	// Initialization
	T.Initialize();

//Assure that the MU used in the initial relaxation stage is k_large
	cudaMemcpyToSymbol(MU, &kl, sizeof(REAL), 0, cudaMemcpyHostToDevice);


#ifdef EQUILIBRATION
	Equilibrate(T, GDOT_OR_SIGMAEXT, t_equilibrate);
#endif


#ifdef FIX_EXT_STRESS


	#ifdef JUSTFLOWCURVE

		#ifdef FROMBELOW

		REAL ss=minstress; 
		while(ss<GDOT_OR_SIGMAEXT){
	  		TimeLoop(T, ss, 0, t_transient, t_run,t_data);
			ss+=stressstep;
	  	}	

		#else 		

		REAL ss=GDOT_OR_SIGMAEXT; 
		while(ss>minstress){
	  		TimeLoop(T, ss, 0, t_transient, t_run,t_data);
			ss-=stressstep;
	  	}	

		#endif	

	#else 		
	
		#ifdef FROMBELOW

		//One assumes force list is ordered from large to small
		for (int i=npoints-1; i>-1; i--){
			REAL ss = (REAL) myNumbers[i];	
	  		TimeLoop(T, ss, 0, t_transient, t_run,t_data);
			if (ss<criticalstress) OcillatoryPhase(T, ss, nofoscilations, ks, kl);
		}

		#else 		

		TimeLoop(T, GDOT_OR_SIGMAEXT, t_equilibrate, t_transient, t_run,t_data);

		for (int i=0; i<npoints; i++){
			REAL ss = (REAL) myNumbers[i];	
	  		TimeLoop(T, ss, 0, t_transient, t_run,t_data);
			if (ss<criticalstress) OcillatoryPhase(T, ss, nofoscilations, ks, kl);
		}
		
		#endif
		
	#endif	

#else


	double ssstress=0.0;
	double ssstress2=0.0;
	unsigned long long int count;
	count=TimeLoop(T, GDOT_OR_SIGMAEXT, t_equilibrate, t_transient, t_run, t_data, ssstress, ssstress2);

	outflow << GDOT_OR_SIGMAEXT << " "<< ssstress << " " << sqrt(ssstress2-ssstress*ssstress)/REAL(sqrt(count)) << " " << ssstress2 << endl;
	
	outflow.close();

	//Print last states
	T.CpyDeviceToHost();
	char filename [256];
	sprintf(filename, "SavedStrain_gdot%6.6f_FF%3.3f_LX%dLY%d_dt%4.4f.dat", GDOT_OR_SIGMAEXT, FF, LX , LY, TIME_STEP);
	ofstream outstrain(filename);
	T.PrintStrain(outstrain);
	outstrain.close();

	sprintf(filename, "SavedForcePoints_gdot%6.6f_FF%3.3f_LX%dLY%d_dt%4.4f.dat", GDOT_OR_SIGMAEXT, FF, LX , LY, TIME_STEP);
	ofstream outfp(filename);
	T.PrintForcePoints(outfp);
	outfp.close();

	sprintf(filename, "SavedForces_gdot%6.6f_FF%3.3f_LX%dLY%d_dt%4.4f.dat", GDOT_OR_SIGMAEXT, FF, LX , LY, TIME_STEP);
	ofstream outforces(filename);
	T.PrintForce(outforces);
	outforces.close();
#endif

	cout << "### ------ CUDA MESO_PLASTIC MODEL------ " << endl;
	// Turn off cronometer
	clock.tac();
	REAL total_time = clock.cpu_elapsed();
	cout << "### Total time    : " << total_time  << " ms" << endl;

	return 0;
}

