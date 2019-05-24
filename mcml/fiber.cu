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
		
		if(NORMAL)
		{
			for (int k = 1; k <= NUM_OF_DETECTOR; k++){
				f[i].radius[k] = collect_r;
				f[i].NA[k] = NAOfDetector;
				f[i].position[k] = collect_r + 2.0 * (collect_r) * (float)(k-1);
				f[i].angle[k] = ANGLE*PI / 180;
				//if(i==0 && k<= NUM_OF_DETECTOR)printf("p[%d][%d]\t%f\n",i,k,f[i].position[k]);
			}


			//f[i].radius[1] = collect_r;					//f[i].radius[1]   = collect_r/2;    //YU-modified         
			//f[i].NA[1]       = NAOfDetector;				
			//f[i].position[1] = 0.0125;                    //first fiber, SDS = 0.03 cm
			//f[i].angle[1]    = ANGLE*PI/180;

			//f[i].radius[2]   = collect_r;          
			//f[i].NA[1]       = NAOfDetector;				
			//f[i].position[2] = 0.0375;                    //second fiber, SDS = 0.04 cm
			//f[i].angle[2]    = ANGLE*PI/180;		
			//
			//f[i].radius[3]   = collect_r;             
			//f[i].NA[3]       = NAOfDetector;				
			//f[i].position[3] = 0.0625;                    //third fiber, SDS = 0.06 cm
			//f[i].angle[3]    = ANGLE*PI/180;

			//f[i].radius[4]   = collect_r;             
			//f[i].NA[4]       = NAOfDetector;				
			//f[i].position[4] = 0.0875;                    //fourth fiber, SDS = 0.08 cm
			//f[i].angle[4]    = ANGLE*PI/180;

			//f[i].radius[5] = collect_r;
			//f[i].NA[5] = NAOfDetector;
			//f[i].position[5] = 0.15;                    //fourth fiber, SDS = 0.08 cm
			//f[i].angle[5] = ANGLE*PI / 180;

			//f[i].radius[6] = collect_r;
			//f[i].NA[6] = NAOfDetector;
			//f[i].position[6] = 0.20;                    //fourth fiber, SDS = 0.08 cm
			//f[i].angle[6] = ANGLE*PI / 180;
		}
		else
		{
			/*
			for (int k = 1; k <= NUM_OF_DETECTOR; k++) {
				f[i].radius[k] = collect_r;
				f[i].NA[k] = NAOfDetector;
				f[i].position[k] = 0.74 + 0.02*(k-1);
				f[i].angle[k] = ANGLE*PI / 180;
			}
			*/
			//¸¯¤ZµØ fitting
			
			f[i].radius[1] = collect_r;      //fiber 1
			f[i].NA[1] = NAOfDetector;
			f[i].position[1] = 0.022;
			f[i].angle[1] = ANGLE*PI / 180;

			f[i].radius[2] = collect_r;      //fuber 2
			f[i].NA[2] = NAOfDetector;
			f[i].position[2] = 0.045;
			f[i].angle[2] = ANGLE*PI / 180;

			f[i].radius[3] = collect_r;      //fiber 3
			f[i].NA[3] = NAOfDetector;
			f[i].position[3] = 0.073;
			f[i].angle[3] = ANGLE*PI / 180;
			
			// DRS with Prof. Sun Chi-Kuang
			
			/*
			f[i].radius[1] = collect_r;      //fiber 1
			f[i].NA[1] = NAOfDetector;
			f[i].position[1] = 0.0215;
			f[i].angle[1] = ANGLE*PI / 180;

			f[i].radius[2] = collect_r;      //fuber 2
			f[i].NA[2] = NAOfDetector;
			f[i].position[2] = 0.041;
			f[i].angle[2] = ANGLE*PI / 180;

			f[i].radius[3] = collect_r;      //fiber 3
			f[i].NA[3] = NAOfDetector;
			f[i].position[3] = 0.061;
			f[i].angle[3] = ANGLE*PI / 180;
			*/

			/*
			f[i].radius[1] = collect_r;      //fiber 1
			f[i].NA[1] = NAOfDetector;
			f[i].position[1] = 0.022;
			f[i].angle[1] = ANGLE*PI / 180;

			f[i].radius[2] = collect_r;      //fuber 2
			f[i].NA[2] = NAOfDetector;
			f[i].position[2] = 0.042;
			f[i].angle[2] = ANGLE*PI / 180;

			f[i].radius[3] = collect_r;      //fiber 3
			f[i].NA[3] = NAOfDetector;
			f[i].position[3] = 0.065;
			f[i].angle[3] = ANGLE*PI / 180;
			*/

			
			f[i].radius[4] = collect_r;      //fiber 4
			f[i].NA[4] = NAOfDetector;
			f[i].position[4] = 0.0215;
			f[i].angle[4] = ANGLE*PI / 180;

			f[i].radius[5] = collect_r;      //fiber 5
			f[i].NA[5] = NAOfDetector;
			f[i].position[5] = 0.041;
			f[i].angle[5] = ANGLE*PI / 180;

			f[i].radius[6] = collect_r;      //fiber 6
			f[i].NA[6] = NAOfDetector;
			f[i].position[6] = 0.061;
			f[i].angle[6] = ANGLE*PI / 180;
			
			/*
			for (int k = 1; k <= NUM_OF_DETECTOR; k++) {
				f[i].radius[k] = collect_r;
				f[i].NA[k] = NAOfDetector;
				f[i].position[k] = fiber1_position + 0.02*(k - 1);
				f[i].angle[k] = ANGLE*PI / 180;
				
			}
			*/
			/*
			f[i].radius[1] = collect_r;      //fiber 1
			f[i].NA[1] = NAOfDetector;
			f[i].position[1] = 0.26;
			f[i].angle[1] = ANGLE*PI / 180;

			f[i].radius[2] = collect_r;      //fuber 2
			f[i].NA[2] = NAOfDetector;
			f[i].position[2] = 0.28;
			f[i].angle[2] = ANGLE*PI / 180;

			f[i].radius[3] = collect_r;      //fiber 3
			f[i].NA[3] = NAOfDetector;
			f[i].position[3] = 0.30;
			f[i].angle[3] = ANGLE*PI / 180;
			
			f[i].radius[4] = collect_r;      //fiber 4
			f[i].NA[4] = NAOfDetector;
			f[i].position[4] = 0.32;
			f[i].angle[4] = ANGLE*PI / 180;
			
			f[i].radius[5] = collect_r;      //fiber 5
			f[i].NA[5] = NAOfDetector;
			f[i].position[5] = 0.34;
			f[i].angle[5] = ANGLE*PI / 180;

			f[i].radius[6] = collect_r;      //fiber 6
			f[i].NA[6] = NAOfDetector;
			f[i].position[6] = 0.36;
			f[i].angle[6] = ANGLE*PI / 180;

			f[i].radius[7] = collect_r;      //fiber 7
			f[i].NA[7] = NAOfDetector;
			f[i].position[7] = 0.38;
			f[i].angle[7] = ANGLE*PI / 180;

			f[i].radius[8] = collect_r;      //fiber 8
			f[i].NA[8] = NAOfDetector;
			f[i].position[8] = 0.40;
			f[i].angle[8] = ANGLE*PI / 180;

			f[i].radius[9] = collect_r;      //fiber 9
			f[i].NA[9] = NAOfDetector;
			f[i].position[9] = 0.42;
			f[i].angle[9] = ANGLE*PI / 180;

			f[i].radius[10] = collect_r;      //fiber 10
			f[i].NA[10] = NAOfDetector;
			f[i].position[10] = 0.44;
			f[i].angle[10] = ANGLE*PI / 180;

			f[i].radius[11] = collect_r;      //fiber 11
			f[i].NA[11] = NAOfDetector;
			f[i].position[11] = 0.46;
			f[i].angle[11] = ANGLE*PI / 180;

			f[i].radius[12] = collect_r;      //fiber 12
			f[i].NA[12] = NAOfDetector;
			f[i].position[12] = 0.48;
			f[i].angle[12] = ANGLE*PI / 180;
			*/
			
			/*
			for(int j = 1; j <= 3; j++)
			{
				f[i].radius[j]   = collect_r;	
				f[i].NA[j]       = NAOfDetector;		
				f[i].position[j] = 0.022*j;	
				f[i].angle[j]    = ANGLE*PI/180;
			}
		

			for(int j = 4; j <= 6; j++)
			{
				f[i].radius[j]   = collect_r;	
				f[i].NA[j]       = NAOfDetector;		
				f[i].position[j] = 0.045*(j-3);	
				f[i].angle[j]    = ANGLE*PI/180;
			}

			for(int j = 7; j <= 9; j++)
			{
				f[i].radius[j]   = collect_r;	
				f[i].NA[j]       = NAOfDetector;		
				f[i].position[j] = 0.073*(j-6);	
				f[i].angle[j]    = ANGLE*PI/180;
			}
			*/
			
		}

	}
}