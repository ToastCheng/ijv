#include "header.h"

void output_fiber(SimulationStruct* sim, float *data, char* output)
{
	ofstream myfile;
	// myfile.open ("GPUMC_output.txt",ios::app); //Wang modified for fitting filename
	myfile.open(output, ios::app);

	double scale1 = (double)0xFFFFFFFF*(double)sim->number_of_photons;
	if(NORMAL)
	{
		for(int i = 0; i < NUM_OF_DETECTOR; i++)
		{
			myfile << double(data[i]/scale1)  << "\t";
		}
	}
	else
	{
		for(int i = 0; i < NUM_OF_DETECTOR; i++)
		{
			myfile << double(data[i]/scale1)  << " ";
		}
	}
	myfile << endl;
    myfile.close();
}


int read_mua_mus(SimulationStruct** simulations, char* input)
{
	// parameters to be modified
	unsigned long number_of_photons = NUMBER_PHOTONS ;  
	const int n_simulations = NUMBER_SIMULATION;
	
	int n_layers = 1;                                   // Wang modified; double layer, default value = 2
	//float medium_n = 1.0;								// float medium_n = 1.33;   // refractive index of medium // YU-modified
	float medium_n = 1.457;  //Wang-modified
	// float lower_thickness = 10.0;						// YU-modified
	float tissue_n = 1.4;                            // refractive index of tissue
	float g_factor = 0.0;                            // anisotropic					//YU-modified

	float start_weight;
	float upper_thickness = 20;

	
	// read the file 
	fstream myfile;
	// myfile.open("GPUMC_input.txt"); //Wang modified for fitting filename
	myfile.open(input);  //Wang modified

	float up_mua[n_simulations];
	float up_mus[n_simulations];


	for (int i = 0; i < n_simulations; i++){
		myfile >> up_mua[i] >> up_mus[i];
		
	}
	myfile.close();


	// Allocate memory for the SimulationStruct array
	*simulations = (SimulationStruct*) malloc(sizeof(SimulationStruct)*n_simulations);
	if(*simulations == NULL){perror("Failed to malloc simulations.\n");return 0;}//{printf("Failed to malloc simulations.\n");return 0;}

	for(int i = 0;i < n_simulations; i++)
	{
		(*simulations)[i].number_of_photons=number_of_photons;
		(*simulations)[i].n_layers = n_layers;

		// Allocate memory for the layers (including one for the upper and one for the lower)
		(*simulations)[i].layers = (LayerStruct*) malloc(sizeof(LayerStruct)*(n_layers+2));
		if((*simulations)[i].layers == NULL){perror("Failed to malloc layers.\n");return 0;}//{printf("Failed to malloc simulations.\n");return 0;}

		// Set upper refractive index (medium)
		(*simulations)[i].layers[0].n = medium_n;	//(*simulations)[i].layers[0].n = medium_n[i]; //YU-modified

		// Set the parameters of tissue (upper layer)
		(*simulations)[i].layers[1].n     = tissue_n;
		(*simulations)[i].layers[1].mua   = up_mua[i];
		(*simulations)[i].layers[1].g = g_factor;			//(*simulations)[i].layers[1].g     = g_factor; //YU-modified 
		(*simulations)[i].layers[1].z_min = 0;
		(*simulations)[i].layers[1].z_max = upper_thickness;	//(*simulations)[i].layers[1].z_max = upper_thickness; //YU-modified
		(*simulations)[i].layers[1].mutr  = 1.0f/(up_mua[i]+up_mus[i]);



		// Set lower refractive index (medium)
		(*simulations)[i].layers[n_layers + 1].n = medium_n;		//(*simulations)[i].layers[n_layers+1].n = medium_n[i]; //YU-modified

		//calculate start_weight
		double n1=(*simulations)[i].layers[0].n;
		double n2=(*simulations)[i].layers[1].n;
		double r = (n1-n2)/(n1+n2);
		r = r*r;
		start_weight = (unsigned int)((double)0xffffffff*(1-r));  
		//start_weight = 1-r;  
		//printf("Start weight=%u\n",start_weight);
		(*simulations)[i].start_weight=start_weight;
	}
	return n_simulations;
}