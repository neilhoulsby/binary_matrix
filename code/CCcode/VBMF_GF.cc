/*	Class to perform VBMF with Gaussian likelihood using method 
	of Raiko et al. 2007

	Author: Neil Houlsby
	Date: 17/6/2012

	NOTE: current speedy implementation with minimial maps requires no 
			rows or columns with zero entries.	
		K = 50 runs out of memory, at start of updateMeans
*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sstream>
#include "strtk.hpp"
//#include <Eigen/Dense>
//#include <Eigen/SVD>

using namespace std;
//using namespace Eigen;

class model {
	public:
		struct data {
			// Vector length N of length 2 vectors of (i, j) values of +1s (rest -1s).
			vector< vector<int> > Xall;
		} data;
		double **mA;
		double **mS;
		double **vA;
		double **vS;
		double *pvA;
		double *pvS;
		double vx;
		double *SmA;
		double *SmS;
		double *SmA2;
		double *SmS2;
		double **SXmA;
		double **SXmS;
		double *SvA;
		double *SvS;
		double gamma;
		double alpha;
		double epsilon;
		
		bool updatePrior;

        double C;
        double time;
        double Niters;

		model(int, int, int, bool);
		double epoch();
		double cost();
		void readData();
		void saveData(int);
		void initialiseReadIn();
		void updateVariances();
		void updateMeans();
		void updateNoise();
		void updatePriorVariances();
		void testGradVariances();
		void testGradMeans(int, int, int);
		void testGradNoise();
		void testGradPriorVariances(int);

	private:
		double Pi;
		int I;
		int J;
		int K;
		void evaluateSums();
		double dot(double*, double*);
		double dotSq(double*, double*);
		double dotX(double**, double**);
};

model::model(int n1, int n2, int n3, bool n4) {
	Pi = atan(1.0) * 4;
	gamma = 0.5;
	alpha = 1;
	epsilon = 1e-4;
	I = n1;
	J = n2;
	K = n3;
	updatePrior = n4;

	// Allocate memory.
	mA = new double*[I];
	vA = new double*[I];
	mS = new double*[J];
	vS = new double*[J];
	for (int i=0; i<I; ++i) {
		mA[i] = new double[K];
		vA[i] = new double[K];
	}
	for (int j=0; j<J; ++j) {
		mS[j] = new double[K];
		vS[j] = new double[K];
	}
	SmA  = new double[K];
	SmS  = new double[K];
	SmA2 = new double[K];
	SmS2 = new double[K];
	SvA  = new double[K];
	SvS  = new double[K];
	SXmA = new double*[K];
	SXmS = new double*[K];
	for (int k=0; k<K; ++k) {
		SXmA[k]  = new double[K];
		SXmS[k]  = new double[K];
	}
	pvA = new double[K];
	pvS = new double[K];
}

void model::readData() {
	// Data comes in as (i, j) pairs, ordered by j.
	ifstream file("../data/XSparse.txt");
	if (!file) { 
		cout << "No Data!\n";
	}
	string line;
	//multimap<int, int> i_to_j, j_to_i;
	while ( file.good() ) {
		vector<int> row;
		getline( file, line, '\n' );
		strtk::parse(line, ",", row);
		if (row.size() > 0) { 
			data.Xall.push_back(row);
	//		i_to_j.insert( pair<int, int>(row[0], row[1]) );
	//		j_to_i.insert( pair<int, int>(row[1], row[0]) );
		}
	}
	file.close();
}

void model::saveData(int timeStamp) {
	stringstream saveLocation;
	saveLocation << "output/paGF_TS" << timeStamp << "_mA.txt";
	ofstream file;
	file.open(saveLocation.str().c_str());
	for (int i = 0; i < I; ++i) {
		for (int k = 0; k < K; ++k) {
			file << mA[i][k] << " ";
		}
		file << "\n";
	}
	file.close();

	saveLocation.str("");
	saveLocation << "output/paGF_TS" << timeStamp << "_mS.txt";
	file.open(saveLocation.str().c_str());
	for (int j = 0; j < J; ++j) {
		for (int k = 0; k < K; ++k) {
			file << mS[j][k] << " ";
		}
		file << "\n";
	}
	file.close();
	
	saveLocation.str("");
	saveLocation << "output/paGF_TS" << timeStamp << "_vA.txt";
	file.open(saveLocation.str().c_str());
	for (int i = 0; i < I; ++i) {
		for (int k = 0; k < K; ++k) {
			file << vA[i][k] << " ";
		}
		file << "\n";
	}
	file.close();
	
	saveLocation.str("");
	saveLocation << "output/paGF_TS" << timeStamp << "_vS.txt";
	file.open(saveLocation.str().c_str());
	for (int j = 0; j < J; ++j) {
		for (int k = 0; k < K; ++k) {
			file << vS[j][k] << " ";
		}
		file << "\n";
	}
	file.close();
	
	saveLocation.str("");
	saveLocation << "output/paGF_TS" << timeStamp << "_pvA.txt";
	file.open(saveLocation.str().c_str());
	for (int k = 0; k < K; ++k) {
		file << pvA[k] << " ";
	}
	file.close();
	
	saveLocation.str("");
	saveLocation << "output/paGF_TS" << timeStamp << "_pvS.txt";
	file.open(saveLocation.str().c_str());
	for (int k = 0; k < K; ++k) {
		file << pvS[k] << " ";
	}
	file.close();
	
	saveLocation.str("");
	saveLocation << "output/paGF_TS" << timeStamp << "_vx.txt";
	file.open(saveLocation.str().c_str());
    file << vx;
    file.close();
	saveLocation.str("");
	saveLocation << "output/paGF_TS" << timeStamp << "_time.txt";
	file.open(saveLocation.str().c_str());
    file << time;
    file.close();
	saveLocation.str("");
	saveLocation << "output/paGF_TS" << timeStamp << "_cost.txt";
	file.open(saveLocation.str().c_str());
    file << C;
    file.close();
	saveLocation.str("");
	saveLocation << "output/paGF_TS" << timeStamp << "_Niters.txt";
	file.open(saveLocation.str().c_str());
    file << Niters;
    file.close();
}

void model::initialiseReadIn() {
	cout << "HERE" << endl;
	ifstream file("../initialisation/mA_init.txt");
	if (!file) { 
		cout << "No Init mA\n";
	}
	string line;
	int i = -1;
	while ( file.good() ) {
		++i;	
		vector<double> row;
		getline( file, line, '\n' );
		strtk::parse(line, ",", row);
		for (int k = 0; k < row.size(); ++k) {
			mA[i][k] = row[k];
		}
	}
	file.close();
	
	file.open("../initialisation/mS_init.txt");
	if (!file) { 
		cout << "No Init mA\n";
	}
	int j = -1;
	while ( file.good() ) {
		++j;	
		vector<double> row;
		getline( file, line, '\n' );
		strtk::parse(line, ",", row);
		for (int k = 0; k < row.size(); ++k) {
			mS[j][k] = row[k];
		}
	}
	file.close();
	
	file.open("../initialisation/vx_init.txt");
	file >> vx;
	file.close();
	for(int i = 0; i<I; ++i) {
		for(int k = 0; k<K; ++k) {
			vA[i][k] = 1;
		}
	}	
	for(int j = 0; j<J; ++j) {
		for(int k = 0; k<K; ++k) {
			vS[j][k] = 1;
		}
	}

    // Initialise Prior uniformly.
    for (int k = 0; k < K; ++k) {
        pvA[k] = 1.0;
        pvS[k] = 1.0;
    }

	// After doing anything recompute the sums.
	evaluateSums();
}

double model::cost() {
	// likeihood - positive terms in X.
	double lik_plus = 0;
	double lik_minus_correction = 0;
	int i, j;
	for (int n=0; n<data.Xall.size(); ++n) {
		i = data.Xall[n][0];
		j = data.Xall[n][1];
		//cout << i << ", " << j << endl;
		lik_plus += log(2 * Pi * vx) / 2
				+ ( pow(1 - dot(mA[i], mS[j]), 2) 	
				+ dotSq(vA[i], mS[j])
				+ dotSq(vS[j], mA[i])
				+ dot(vA[i], vS[j]) ) / (2 * vx);
		lik_minus_correction += log(2 * Pi * vx) / 2
				+ ( pow(-1 - dot(mA[i], mS[j]), 2) 	
				+ dotSq(vA[i], mS[j])
				+ dotSq(vS[j], mA[i])
				+ dot(vA[i], vS[j]) ) / (2 * vx);
	}
	
	// likelihood - negative terms.
	double lik_minus = (I * J) * log(2 * Pi * vx) / 2
					+ ( I * J
					+ 2 * dot(SmA, SmS)
					+ dotX(SXmA, SXmS)
					+ dot(SvA, SmS2)
					+ dot(SvS, SmA2)
					+ dot(SvA, SvS) ) / (2 * vx);
		
	// priors
	double priorA = 0;	
	for (int i=0; i<I; ++i) {
		for (int k=0; k<K; ++k) {
			priorA += - 0.5 
					- log(vA[i][k] / pvA[k]) / 2
					+ ( pow(mA[i][k], 2) + vA[i][k]  ) / (2 * pvA[k]);
		}
	} 
	double priorS = 0;
	for (int j=0; j<J; ++j) {
		for (int k=0; k<K; ++k) {
			priorS += - 0.5 
					- 0.5 * log(vS[j][k] / pvS[k])
					+ ( pow(mS[j][k], 2) + vS[j][k]  ) / (2 * pvS[k]);
		}
	}
//	cout << "min " << lik_minus << endl;
//	cout << "corr " << lik_minus_correction << endl;
//	cout << "plus " << lik_plus << endl;
	return lik_minus - lik_minus_correction
			 + lik_plus + priorA + priorS;
}

void model::updateVariances() {
	// Gradients seem OK - although where to evaluateSums() is important.
	for (int k=0; k<K; ++k) {
		double term_minus_all = (SmS2[k] + SvS[k]) / vx;
		term_minus_all = pow( (1.0 / pvA[k]) + term_minus_all, -1);
		for (int i=0; i<I; ++i) {
			vA[i][k] = term_minus_all;
		}
	}
	evaluateSums();
	for (int k=0; k<K; ++k) {
		double term_minus_all = (SmA2[k] + SvA[k]) / vx;
		term_minus_all = pow( (1.0 / pvS[k]) + term_minus_all, -1);
		for (int j=0; j<J; ++j) {
			vS[j][k] = term_minus_all;
		}
	}
	evaluateSums();
}

void model::updateMeans() {
	// Gradient problems when vals get large - getting instability 
	// non-small gamma.
	double termA_plus[I][K];
	double termS_plus[J][K];
	double termA_corr[I][K];
	double termS_corr[J][K];
	double dCdA[I][K];
	double dCdS[J][K];
	double d2CdA2[I][K];
	double d2CdS2[J][K];
	// some pre-computations
	for (int k=0; k<K; ++k) {
		for (int i=0; i<I; ++i) {
			termA_plus[i][k] = 0;
			termA_corr[i][k] = 0;
		}
		for (int j=0; j<J; ++j) {
			termS_plus[j][k] = 0;
			termS_corr[j][k] = 0;
		}
	}
	for (int n=0; n<data.Xall.size(); ++n) {
		int i = data.Xall[n][0];
		int j = data.Xall[n][1];
		for (int k=0; k<K; ++k) {
			// TODO: prob could save a bunch of multiple calcs here.
			double pred = dot(mA[i], mS[j]);
			termA_plus[i][k] += mA[i][k] / pvA[k]
							+ ( - (1.0 - pred) * mS[j][k] 
							+ mA[i][k] * vS[j][k] ) / vx;
			termA_corr[i][k] += mA[i][k] / pvA[k]
							+ ( - (-1.0 - pred) * mS[j][k] 
							+ mA[i][k] * vS[j][k] ) / vx;
			termS_plus[j][k] += mS[j][k] / pvS[k]
							+ ( - (1.0 - pred) * mA[i][k] 
							+ mS[j][k] * vA[i][k] ) / vx;
			termS_corr[j][k] += mS[j][k] / pvS[k]
							+ ( - (-1.0 - pred) * mA[i][k] 
							+ mS[j][k] * vA[i][k] ) / vx;
		}
	}
	double xterm;
	double term_minus;
	// Do loop this way round for cross-term efficiency.
	for (int i=0; i<I; ++i) {
		for (int k=0; k<K; ++k) {
			xterm = 0;
			for (int kd=0; kd<K; ++kd) {
				xterm += mA[i][kd]*SXmS[k][kd];
			}
			term_minus = mA[i][k] / pvA[k]
						+ ( SmS[k] + xterm + mA[i][k]*SvS[k] ) / vx;
			dCdA[i][k] = term_minus - termA_corr[i][k] + termA_plus[i][k];
			d2CdA2[i][k] = pow(vA[i][k], -1);
		}
	}
	
	//cout << "tap = " << termA_plus[I-1][K-1] << endl;
	//cout << "tac = " << termA_corr[I-1][K-1] << endl;
	//cout << "tam = " << term_minus << endl;
	//cout << "analytic = " << dCdA[I-1][K-1] << endl;
	//testGradMeans(I-1, J-1, K-1);
	
	for (int j=0; j<J; ++j) {
		for (int k=0; k<K; ++k) {
			xterm = 0;
			for (int kd=0; kd<K; ++kd) {
				xterm += mS[j][kd]*SXmA[kd][k];
			}
			term_minus = mS[j][k] / pvS[k]
						+ ( SmA[k] + xterm + mS[j][k]*SvA[k] ) / vx;
			dCdS[j][k] = term_minus - termS_corr[j][k] + termS_plus[j][k];
			d2CdS2[j][k] = pow(vS[j][k], -1);
		}
	}
	for (int k=0; k<K; ++k) {
		for (int i=0; i<I; ++i) {
			mA[i][k] = mA[i][k] - gamma*pow(d2CdA2[i][k], -alpha) * dCdA[i][k];
		}	
		for (int j=0; j<J; ++j) {
			mS[j][k] = mS[j][k] - gamma*pow(d2CdS2[j][k], -alpha) * dCdS[j][k];
		}
	}
	evaluateSums();	
}

void model::updateNoise() {
	// Update same as MATLAB case, numerical grads a little off, prob unstable.
	double term_plus = 0;	
	double term_corr = 0;	
	for (int n=0; n<data.Xall.size(); ++n) {
		int i = data.Xall[n][0];
		int j = data.Xall[n][1];
		term_plus += ( pow(1 - dot(mA[i], mS[j]), 2)
					+ dotSq(vA[i], mS[j])
					+ dotSq(vS[j], mA[i])
					+ dot(vA[i], vS[j]) ) / (I * J);
		term_corr += ( pow(-1 - dot(mA[i], mS[j]), 2)
					+ dotSq(vA[i], mS[j])
					+ dotSq(vS[j], mA[i])
					+ dot(vA[i], vS[j]) ) / (I * J);
	}
	double	term_minus = (I * J  +  2.0 * dot(SmA, SmS)
						+ dotX(SXmA, SXmS)
						+ dot(SvA, SmS2)
						+ dot(SvS, SmA2)
						+ dot(SvA, SvS) ) / (I * J);
	vx = term_plus - term_corr + term_minus;
}

void model::updatePriorVariances() {
    for (int k = 0; k < K; ++k) {
        double sum = 0;
        for (int i = 0; i < I; ++i) {
            sum += pow(mA[i][k], 2) + vA[i][k];
        }
        pvA[k] = sum / I;
    }
//  for (int k = 0; k < K; ++k) {
//      double sum = 0;
//      for (int j = 0; j < J; ++j) {
//          sum += pow(mS[j][k], 2) + vS[j][k];
//      }
//      pvS[k] = sum / J;
//  }
    //testGradPriorVariances(0);
}


void model::testGradVariances() {
	double pert = 1e-6;
	int i = 0;
	int j = 0;
	int k = 0;
	vA[i][k] += pert;
	evaluateSums();	
	double Cplus = cost();
	vA[i][k] -= 2 * pert;
	evaluateSums();	
	double Cminus = cost();
	cout << "vA[" << i << "][" << k << "] = " << vA[i][k] + pert << endl;
	cout << "dCdvA[" << i << "][" << k << "] = " << (Cplus - Cminus) / (2 * pert) << endl;
	vA[i][k] += pert;
	vS[j][k] += pert;
	evaluateSums();	
	Cplus = cost();
	vS[j][k] -= 2 * pert;
	evaluateSums();	
	Cminus = cost();
	cout << "vS[" << j << "][" << k << "] = " << vS[j][k] + pert << endl;
	cout << "dCdvS[" << j << "][" << k << "] = " << (Cplus - Cminus) / (2 * pert) << endl;
	vS[j][k] += pert;
	evaluateSums();
}

void model::testGradMeans(int i, int j, int k) {
	double pert = 1e-4;
	mA[i][k] += pert;
	evaluateSums();
	double Cplus = cost();
	mA[i][k] -= 2 * pert;
	evaluateSums();
	double Cminus = cost();
	cout << "mA[" << i << "][" << k << "] = " << mA[i][k] + pert << endl;
	cout << "dCdmA[" << i << "][" << k << "] = " << (Cplus - Cminus) / (2 * pert) << endl;
	mA[i][k] += pert;
	mS[j][k] += pert;
	evaluateSums();
	Cplus = cost();
	mS[j][k] -= 2 * pert;
	evaluateSums();
	Cminus = cost();
	mS[j][k] += pert;
	cout << "mS[" << j << "][" << k << "] = " << mS[j][k] + pert << endl;
	cout << "dCdmS[" << j << "][" << k << "] = " << (Cplus - Cminus) / (2 * pert) << endl;
}

void model::testGradNoise() {
	double pert = 1e-5;
	vx += pert;
	double Cplus = cost();
	vx -= 2 * pert;
	double Cminus = cost();
	cout << "vx = " << vx + pert << endl;
	cout << "dCdvx = " << (Cplus - Cminus) / (2 * pert) << endl;
}

void model::testGradPriorVariances(int k) {
    double pert = 1e-6;
    pvA[k] += pert;
    double Cplus = cost();
    pvA[k] -= 2 * pert;
    double Cminus = cost();
    cout << "pvA[" << k << "] = " << pvA[k] + pert << endl;
    cout << "dCdpvA[" << k << "] = " << (Cplus - Cminus) / (2 * pert) << endl;
    pvA[k] += pert;
    pvS[k] += pert;
    Cplus = cost();
    pvS[k] -= 2 * pert;
    Cminus = cost();
    pvS[k] += pert;
    cout << "pvS[" << k << "] = " << pvS[k] + pert << endl;
    cout << "dCdpvS[" << k << "] = " << (Cplus - Cminus) / (2 * pert) << endl;
}


void model::evaluateSums() {
	for (int k=0; k<K; ++k) {
		SmA[k] = 0;
		SmS[k] = 0;
		SmA2[k] = 0;
		SmS2[k] = 0;
		SvA[k] = 0;
		SvS[k] = 0;
		for (int i=0; i<I; ++i) {
			SmA[k]  += mA[i][k];
			SmA2[k] += pow(mA[i][k], 2);
			SvA[k]  += vA[i][k];
		}
		for (int j=0; j<J; ++j) {
			SmS[k]  += mS[j][k];
			SmS2[k] += pow(mS[j][k], 2);
			SvS[k]  += vS[j][k];
		}
		for (int kd=0; kd<K; ++kd) {
			SXmA[k][kd] = 0;
			SXmS[k][kd] = 0;
			for (int i=0; i<I; ++i) {
				SXmA[k][kd] += mA[i][k] * mA[i][kd];
			}
			for (int j=0; j<J; ++j) {
				SXmS[k][kd] += mS[j][k] * mS[j][kd];
			}
		}
	}
}

double model::dot(double *p, double *q) {
	// Calculates dot product between two vectors of length k.
	double sum = 0;
	for (int k = 0; k<K; ++k) {
		sum += p[k] * q[k];
	}
	return sum;
}

double model::dotSq(double *p, double *q) {
	// Calculates dot product between two vectors of length K, 
	// squaring each element of the second.
	double sum = 0;
	for (int k = 0; k<K; ++k) {
		sum += p[k] * pow(q[k], 2);
	}
	return sum;
}

double model::dotX(double **p, double **q) {
	// Calculates sum of elementwise products of two K*K arrays. 
	double sum = 0;
	for (int k = 0; k<K; ++k) {
		for (int kd = 0; kd<K; ++kd) {
			sum += p[k][kd] * q[k][kd];
		}
	}
	return sum;
}

double model::epoch(){
	// Sequence for perfoming optimisation
	updateVariances();
	// cout << "Cinter = " << cost() << endl;
	updateMeans();
	// cout << "Cinter = " << cost() << endl;
	updateNoise();
	if (updatePrior) { updatePriorVariances(); }
	return cost();
}

main(int argc, char *argv[]) {
	int I, J, K;
	bool updatePrior;
	// Check number of CL arguments.
	if ( argc != 5 ) {
		cout << "need I, J, K, updatePrior\n";
	} else {
		string tmp = argv[1];
		I = atoi(tmp.c_str());
		tmp = argv[2];
		J = atoi(tmp.c_str());
		tmp = argv[3];
		K = atoi(tmp.c_str());
        tmp = argv[4];
        updatePrior = (strcasecmp (tmp.c_str(), "true") == 0 ||
                            atoi(tmp.c_str()) != 0);
 	}

	model pa(I, J, K, updatePrior);
	cout << "MODEL CREATED\n";
	pa.readData();
	cout << "DATA READ IN\n";
	pa.initialiseReadIn();
	cout << "MODEL INITIALISED\n";
	clock_t init, final;
	init = clock();	
	double C;
	double Cold = 1e10;
	bool converged = false;
	int iter = 0;
	cout << "INIT COST = " << pa.cost() << "\n";
	int timeStamp = 1;
	while ( iter < 3 || (!converged && iter < 250) ) {
		C = pa.epoch();
		if (iter%10 == 0) {
			pa.C = C;
			pa.Niters = iter;
			final = clock() - init;
   			pa.time = (double)final / (double)CLOCKS_PER_SEC;
			pa.saveData(timeStamp);	
			++timeStamp;
			cout << "Iter: " << iter << ", cost = " << C << endl;
		}
		if (abs(Cold - C) < pa.epsilon * abs(C)) converged = true;
		if (Cold > C) {
			pa.gamma *= 1.1;
		} else {
			pa.gamma *= 0.5;
		}
		Cold = C;
		++iter;
	}
    pa.C = C;
	pa.Niters = iter;
	final = clock() - init;
    pa.time = (double)final / (double)CLOCKS_PER_SEC;
	pa.saveData(timeStamp);
	ofstream file;	
	file.open("output/paGF_numTSs.txt");
	file << timeStamp;
	file.close(); 
	
	cout << "SOLVED IN " << pa.time << "s\n";

	return 0;
}
