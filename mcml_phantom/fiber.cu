#include "header.h"

void fiber_initialization(Fibers* f/*, float fiber1_position*/)  //Wang modified
{
	for(int i = 0; i < NUM_THREADS; i++)
	{
	    for(int j = 0; j <= NUM_OF_DETECTOR; j++)
			f[i].data[j] = 0;
		
		f[i].radius[0]   = illumination_r;          // source fiber			
	    f[i].NA[0]       = NAOfSource;				
	    f[i].angle[0]    = ANGLE*PI/180;
	    f[i].position[0] = 0.0;			
			
		f[i].radius[1] = collect_r;      //fiber 1
		f[i].NA[1] = NAOfDetector;
		f[i].position[1] = 0.31;
		f[i].angle[1] = ANGLE*PI / 180;

		f[i].radius[2] = collect_r;      //fuber 2
		f[i].NA[2] = NAOfDetector;
		f[i].position[2] = 0.4;
		f[i].angle[2] = ANGLE*PI / 180;

		f[i].radius[3] = collect_r;      //fiber 3
		f[i].NA[3] = NAOfDetector;
		f[i].position[3] = 0.44;
		f[i].angle[3] = ANGLE*PI / 180;
		
		
		f[i].radius[4] = collect_r;      //fiber 4
		f[i].NA[4] = NAOfDetector;
		f[i].position[4] = 0.53;
		f[i].angle[4] = ANGLE*PI / 180;

		f[i].radius[5] = collect_r;      //fiber 5
		f[i].NA[5] = NAOfDetector;
		f[i].position[5] = 0.63;
		f[i].angle[5] = ANGLE*PI / 180;

		

	}
}