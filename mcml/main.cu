#include "header.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


void FreeSimulationStruct(SimulationStruct* sim, int n_simulations);
int read_mua_mus(SimulationStruct** simulations, char* input);
void DoOneSimulation(SimulationStruct* simulation, int index, char* output);




int main(int argc,char* argv[])
{
	SimulationStruct* simulations;
	int n_simulations;
	unsigned long long seed = (unsigned long long) time(NULL);// Default, use time(NULL) as seed
	
	n_simulations = read_mua_mus(&simulations, argv[1]);

	if(n_simulations == 0)
	{
		printf("Something wrong with read_simulation_data!\n");
		return 1;
	}


	clock_t time1,time2;
	
	// Start the clock
    time1=clock();

	//perform all the simulations
	for(int i = 0; i < n_simulations; i++)
	{
		// Run a simulation
		printf("simulating %d\n",i);
		DoOneSimulation(&simulations[i],i, argv[2]);
	}

	time2=clock();
	printf("Simulation time: %.2f sec\n",(double)(time2-time1)/CLOCKS_PER_SEC);

	FreeSimulationStruct(simulations, n_simulations);
	
	//system("PAUSE");
	return 0; 
}

void FreeSimulationStruct(SimulationStruct* sim, int n_simulations)
{
	for(int i = 0;i < n_simulations; i++) free(sim[i].layers);
	free(sim);
}