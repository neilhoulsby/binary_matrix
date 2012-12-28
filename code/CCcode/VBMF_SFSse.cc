/*	Class to perform VBMF with Gaussian likelihood using method 
	of Raiko et al. 2007

	Author: Neil Houlsby
	Date: 17/6/2012
	
	Command line arguments:
		1. int I
		2. int J
		3. int K
		4. int num samples from matrix
		5. int num samples from rows
		6. int num samples from cols
		7. bool whether to update the prior
                     (across latent dimension)
		8. bool whether to include local offsets

*/

// Aligns with VBMF_SJSpFSs_CppVersion.m
// TODO: will say if passed in I, J too large, will segfault if too small.
// TODO: make matrix samples intersection of row and col samples?
// TODO: can improve d2CdA2.

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "strtk.hpp"
#include <set>
#include <sstream>

using namespace std;

class model {
	public:
		struct data {
			// Vector length N of length 2 vectors of (i, j) values of +1s (rest -1s).
			vector< vector<int> > X;
			// So that one does not sample +1s.
			set< vector<int> > Xset;
			// Fitting params for each +1.
			vector<double> Z;
		} data;
		struct samples {
			// N.b. Outer structure could be a map.
			vector< vector<int> > matrix;
			// I sets of samples (ints) for each row.
			vector< vector<int> > rows;
			// J sets of samples (ints) for each row.
			vector< vector<int> > cols;
			// Fitting parameters.
			vector<double> Zmatrix;
			vector< vector<double> > Zrows;
			vector< vector<double> > Zcols;
		} samples;
		double **mA;
		double **mS;
		double mb;
		double **vA;
		double **vS;
		double vb;

		double *pvA;
		double *pvS;
		double pvb;

		double gamma;
		double alpha;
		double epsilon;
		bool localSparsity;
		bool updatePrior;
	
        double C;
        double time;
        double Niters;

		model(int, int, int, int, int, int, bool, bool);
		double epoch();
		double cost();
		void readData();
		void saveData(int);
		void subSample();
		void initialiseReadIn();
		void updateVariances();
		void updateMeans();
		void updateOffset();
		void updatePriorVariances();
		void updateFittingParams();
		void testGradVariances();
		void testGradMeans(int, int, int);
		void testGradNoise();
		void testGradPriorVariances(int);

	private:
		double Pi;
		int I;
		int J;
		int K;
		int NSSmatrix;
		int NSSrows;
		int NSScols;
		double commonTerm(int, int);
		double dot(double*, double*);
		double dotSq(double*, double*);
		double dotX(double**, double**);
		double sigma(double);
		double lambda(double);
};

model::model(int p1, int p2, int p3, int p4, int p5, int p6, bool p7, bool p8) {
	// Stuff Read in from CL.
	I = p1;
	J = p2;
	K = p3;
	NSSmatrix = p4;
	NSSrows = p5;
	NSScols = p6;
	localSparsity = p7;
	updatePrior = p8;
	if (localSparsity) {K += 2;}

	// Parameters.
	Pi = atan(1.0) * 4;
	gamma = 0.5;
	alpha = 1;
	epsilon = 1e-5;

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
	int max_i = 0;
	int max_j = 0;
	while ( file.good() ) {
		vector<int> row;
		getline( file, line, '\n' );
		strtk::parse(line, ",", row);
		if (row.size() > 0) { 
			data.X.push_back(row);
			data.Xset.insert(row);
			if (row[0] > max_i) { max_i = row[0]; } 
			if (row[1] > max_j) { max_j = row[1]; } 
		}
	}
	file.close();
	if (max_i != I - 1) { cout << "I wrong!" << " I = " << max_i << endl; }
	if (max_j != J - 1) { cout << "J wrong!" << " J = " << max_j << endl; }
}


void model::subSample() {
	samples.matrix.clear();	
	samples.rows.clear();	
	samples.cols.clear();	
	set< vector<int> > matrixSamplesSet;
	int itSample = 0;
	while (itSample < NSSmatrix) {
		int iSample = (int)(I * (double)(rand()) / RAND_MAX);
		int jSample = (int)(J * (double)(rand()) / RAND_MAX);
		vector<int> sample;
		sample.push_back(iSample);
		sample.push_back(jSample);
		if ( (matrixSamplesSet.find(sample) == matrixSamplesSet.end())
				&& (data.Xset.find(sample) == data.Xset.end()) ) {
			samples.matrix.push_back(sample);
			matrixSamplesSet.insert(sample);
			++itSample;
		}
		if (itSample == (I*J - data.X.size()) ) {
			cout << "Sampled all -1s from matrix, adjusting NSSmatrix\n";
			NSSmatrix = itSample;
		}
	}
	set< int > rowSamplesSet;
	vector< int > rowSamples;
	for (int i = 0; i < I; ++i)	{
		itSample = 0;
		while (itSample < NSSrows) {
			int jSample = (int)(J * (double)(rand()) / RAND_MAX);
			vector<int> sample;
			sample.push_back(i);
			sample.push_back(jSample);
			if ( rowSamplesSet.find(jSample) == rowSamplesSet.end()
					&& data.Xset.find(sample) == data.Xset.end() ) {
				rowSamples.push_back(jSample);
				rowSamplesSet.insert(jSample);
				++itSample;
			}
			// To check if we are trying to oversample a row, need to record num +1s per row.
			// Currently, it will just loop forever.
		}
		samples.rows.push_back(rowSamples);
		rowSamples.clear();
		rowSamplesSet.clear();
	}
	set< int > colSamplesSet;
	vector< int > colSamples;
	for (int j = 0; j < J; ++j)	{
		itSample = 0;
		while (itSample < NSScols) {
			int iSample = (int)(I * (double)(rand()) / RAND_MAX);
			vector<int> sample;
			sample.push_back(iSample);
			sample.push_back(j);
			if ( colSamplesSet.find(iSample) == colSamplesSet.end()
					&& data.Xset.find(sample) == data.Xset.end() ) {
				colSamples.push_back(iSample);
				colSamplesSet.insert(iSample);
				++itSample;
			}
		}
		samples.cols.push_back(colSamples);
		colSamples.clear();
		colSamplesSet.clear();
	}
}


void model::saveData(int timeStamp) {
	stringstream saveLocation;
	saveLocation << "output/paSFSse_TS" << timeStamp << "_mA.txt";
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
	saveLocation << "output/paSFSse_TS" << timeStamp << "_mS.txt";
	file.open(saveLocation.str().c_str());
	for (int j = 0; j < J; ++j) {
		for (int k = 0; k < K; ++k) {
			file << mS[j][k] << " ";
		}
		file << "\n";
	}
	file.close();
	
	saveLocation.str("");
	saveLocation << "output/paSFSse_TS" << timeStamp << "_vA.txt";
	file.open(saveLocation.str().c_str());
    for (int i = 0; i < I; ++i) {
        for (int k = 0; k < K; ++k) {
            file << vA[i][k] << " ";
        }
        file << "\n";
    }
    file.close();

	saveLocation.str("");
	saveLocation << "output/paSFSse_TS" << timeStamp << "_vS.txt";
	file.open(saveLocation.str().c_str());
    for (int j = 0; j < J; ++j) {
        for (int k = 0; k < K; ++k) {
            file << vS[j][k] << " ";
        }
        file << "\n";
    }
    file.close();

	saveLocation.str("");	
	saveLocation << "output/paSFSse_TS" << timeStamp << "_pvA.txt";
	file.open(saveLocation.str().c_str());
    for (int k = 0; k < K; ++k) {
        file << pvA[k] << " ";
    }
    file.close();

	saveLocation.str("");	
	saveLocation << "output/paSFSse_TS" << timeStamp << "_pvS.txt";
	file.open(saveLocation.str().c_str());
    for (int k = 0; k < K; ++k) {
        file << pvS[k] << " ";
    }
    file.close();
	
	saveLocation.str("");	
	saveLocation << "output/paSFSse_TS" << timeStamp << "_mb.txt";
	file.open(saveLocation.str().c_str());
	file << mb;
	file.close();
	saveLocation.str("");	
	saveLocation << "output/paSFSse_TS" << timeStamp << "_vb.txt";
	file.open(saveLocation.str().c_str());
	file << vb;
	file.close();
	saveLocation.str("");	
	saveLocation << "output/paSFSse_TS" << timeStamp << "_pvb.txt";
	file.open(saveLocation.str().c_str());
	file << pvb;
	file.close();
	saveLocation.str("");	
	saveLocation << "output/paSFSse_TS" << timeStamp << "_time.txt";
	file.open(saveLocation.str().c_str());
    file << time;
    file.close();
	saveLocation.str("");	
	saveLocation << "output/paSFSse_TS" << timeStamp << "_cost.txt";
	file.open(saveLocation.str().c_str());
    file << C;
    file.close();
	saveLocation.str("");	
	saveLocation << "output/paSFSse_TS" << timeStamp << "_NSSrows.txt";
	file.open(saveLocation.str().c_str());
    file << NSSrows;
    file.close();
	saveLocation.str("");	
	saveLocation << "output/paSFSse_TS" << timeStamp << "_NSScols.txt";
	file.open(saveLocation.str().c_str());
    file << NSScols;
    file.close();
	saveLocation.str("");	
	saveLocation << "output/paSFSse_TS" << timeStamp << "_NSSmatrix.txt";
	file.open(saveLocation.str().c_str());
    file << NSSmatrix;
    file.close();
	saveLocation.str("");	
	saveLocation << "output/paSFSse_TS" << timeStamp << "_Niters.txt";
	file.open(saveLocation.str().c_str());
    file << Niters;
    file.close();
}

void model::initialiseReadIn() {
	// Initialise mean parameters from file.
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
		if (localSparsity && row.size() > 0) {
			mA[i][K-2] = 1;
			mA[i][K-1] = 0;
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
		if (localSparsity && row.size() > 0) {
			mS[j][K-1] = 1;
			mS[j][K-2] = 0;
		}
	}
	file.close();
	
	// Initialise offset from file.
	file.open("../initialisation/mb_init.txt");
	file >> mb;
	file.close();

	// Initialise variances uniformly.
	for(int i = 0; i<I; ++i) {
		for(int k = 0; k<K; ++k) {
			if (localSparsity && k == K - 2) {
				vA[i][k] = 1.0e-10;
			} else {	
				vA[i][k] = 1.0;
			}
		}
	}	
	for(int j = 0; j<J; ++j) {
		for(int k = 0; k<K; ++k) {
			if (localSparsity && k == K - 1) {
				vS[j][k] = 1.0e-10;
			} else {	
				vS[j][k] = 1.0;
			}
		}
	}
	vb  = 1.0;

	// Initialise Prior uniformly, large variance on local sparsity param, small on fixed row or ones.
	for (int k = 0; k < K; ++k) {
		if (localSparsity && k == K - 1) {
			pvA[k] = 1.0e10;
			pvS[k] = 1.0;
		} else if (localSparsity && k == K - 2) {
			pvA[k] = 1.0;
			pvS[k] = 1.0e10;
		} else {
			pvA[k] = 1.0;
			pvS[k] = 1.0;
		}
	}
	pvb = 1.0e10;
	
	// Fill data.Z and samples.Zmatrix/Zrows/Zcols with zeros for time being.
	for(int n = 0; n < data.X.size(); ++n) {
		data.Z.push_back(0.0);
	}
	for(int n = 0; n < samples.matrix.size(); ++n) {
		samples.Zmatrix.push_back(0.0);
	}
	for(int i = 0; i < I; ++i) {
		vector<double> tmp;
		for(int j = 0; j < samples.rows[i].size(); ++j) {
			tmp.push_back(0.0);
		}
		samples.Zrows.push_back(tmp);
	}	
	for(int j = 0; j < J; ++j) {
		vector<double> tmp;
		for(int i = 0; i < samples.cols[j].size(); ++i) {
			tmp.push_back(0.0);
		}
		samples.Zcols.push_back(tmp);
	}
}

double model::cost() {
	// Likeihood - positive terms in X.
	// N.b. don't need correction term because approx sum only over -ve elements.
	double lik_plus = 0;
	int i, j;
	for (int n = 0; n < data.X.size(); ++n) {
		i = data.X[n][0];
		j = data.X[n][1];
		lik_plus += - 0.5 * (dot(mA[i], mS[j]) + mb)
					+ 0.5 * data.Z[n] - log(sigma(data.Z[n])) - lambda(data.Z[n])
					* ( commonTerm(i, j) - data.Z[n] * data.Z[n] ); 
	}
	// likelihood - negative terms
	double sum = 0;
	for (int n = 0; n < samples.matrix.size(); ++n) {
		i = samples.matrix[n][0];
		j = samples.matrix[n][1];
		sum += 0.5 * (dot(mA[i], mS[j]) + mb)
				+ 0.5 * samples.Zmatrix[n] - log(sigma(samples.Zmatrix[n])) 
				- lambda(samples.Zmatrix[n])
				* ( commonTerm(i, j) - samples.Zmatrix[n] * samples.Zmatrix[n] ); 
	}
	double lik_minus = sum * I * J / NSSmatrix; 
		
	// priors
	double priorA = 0;	
	for (int i = 0; i < I; ++i) {
		for (int k = 0; k < K; ++k) {
			priorA += - 0.5 
					- 0.5 * log(vA[i][k] / pvA[k])
					+ ( pow(mA[i][k], 2) + vA[i][k]  ) / (2 * pvA[k]);
		}
	} 
	double priorS = 0;
	for (int j = 0; j < J; ++j) {
		for (int k = 0; k < K; ++k) {
			priorS += - 0.5 
					- 0.5 * log(vS[j][k] / pvS[k])
					+ ( pow(mS[j][k], 2) + vS[j][k]  ) / (2 * pvS[k]);
		}
	}
	double priorb = 0.5 * ( ( mb*mb + vb ) / pvb - log(vb / pvb)  - 1.0);

	return lik_minus  + lik_plus + priorA + priorS + priorb;
}

void model::updateVariances() {
	double termsA_plus[I][K];
	double termsA_minus[I][K];
	double termsS_plus[J][K];
	double termsS_minus[J][K];
    for (int k=0; k<K; ++k) {
        for (int i=0; i<I; ++i) {
            termsA_plus[i][k] = 0;
            termsA_minus[i][k] = 0;
        }
        for (int j=0; j<J; ++j) {
            termsS_plus[j][k] = 0;
            termsS_minus[j][k] = 0;
        }
    }
	for (int k = 0; k < K; ++k) {
		for (int n = 0; n < data.X.size(); ++n) {
		 	int i = data.X[n][0];	
		 	int j = data.X[n][1];	
			termsA_plus[i][k] += - 2.0 * lambda(data.Z[n]) 
							* ( mS[j][k] * mS[j][k] + vS[j][k] ); 
		}
		for (int i = 0; i < I; ++i) {
			for (int itrow = 0; itrow < samples.rows[i].size(); ++itrow) {
				int j = samples.rows[i][itrow];
				termsA_minus[i][k] += - 2.0 * lambda(samples.Zrows[i][itrow]) 
										* ( mS[j][k] * mS[j][k] + vS[j][k] ); 
			}
		}
		for (int i = 0; i < I; ++i) {
			termsA_minus[i][k] *= J / NSSrows;
			if (localSparsity && k == K - 2) {
				vA[i][k] = 1.0e-10;
			} else {
				vA[i][k] = 1.0 / (1.0 / pvA[k] + termsA_plus[i][k] + termsA_minus[i][k]);
			}
		}
	}
	for (int k = 0; k < K; ++k) {
		for (int n = 0; n < data.X.size(); ++n) {
		 	int i = data.X[n][0];	
		 	int j = data.X[n][1];	
			termsS_plus[j][k] += - 2.0 * lambda(data.Z[n]) 
							* ( mA[i][k] * mA[i][k] + vA[i][k] ); 
		}
		for (int j = 0; j < J; ++j) {
			for (int itcol = 0; itcol < samples.cols[j].size(); ++itcol) {
				int i = samples.cols[j][itcol];
				termsS_minus[j][k] += - 2.0 * lambda(samples.Zcols[j][itcol]) 
										* ( mA[i][k] * mA[i][k] + vA[i][k] ); 
			}
		}
		for (int j = 0; j < J; ++j) {
			termsS_minus[j][k] *= I / NSScols;
			if (localSparsity && k == K - 1) {
				vS[j][k] = 1.0e-10;
			} else {
				vS[j][k] = 1.0 / (1.0 / pvS[k] + termsS_plus[j][k] + termsS_minus[j][k]);
			}
		}
	}
}


void model::updateMeans() {
	double termA_plus[I][K];
	double termS_plus[J][K];
	double termA_minus[I][K];
	double termS_minus[J][K];
	double dCdA[I][K];
	double dCdS[J][K];
	for (int k = 0; k < K; ++k) {
		for (int i = 0; i < I; ++i) {
			termA_plus[i][k] = 0;
			termA_minus[i][k] = 0;
		}
		for (int j = 0; j < J; ++j) {
			termS_plus[j][k] = 0;
			termS_minus[j][k] = 0;
		}
	}
	for (int k = 0; k < K; ++k) {
		for (int n = 0; n < data.X.size(); ++n) {
			int i = data.X[n][0];
			int j = data.X[n][1];
			termA_plus[i][k] += - 0.5 * mS[j][k] - 2.0 * lambda(data.Z[n])
								* ( mA[i][k] * vS[j][k] 
								+ mS[j][k] * dot(mA[i], mS[j]) 
								+ mb * mS[j][k] );
			termS_plus[j][k] += - 0.5 * mA[i][k] - 2.0 * lambda(data.Z[n])
								* ( mS[j][k] * vA[i][k] 
								+ mA[i][k] * dot(mA[i], mS[j]) 
								+ mb * mA[i][k] );
			
		}
		for (int i = 0; i < I; ++i) {
			for (int itrow = 0; itrow < samples.rows[i].size(); ++itrow) {
				int j = samples.rows[i][itrow];
				termA_minus[i][k] += 0.5 * mS[j][k] - 2.0 * lambda(samples.Zrows[i][itrow])
									* ( mA[i][k] * vS[j][k] 
									+ mS[j][k] * dot(mA[i], mS[j]) 
									+ mb * mS[j][k] );
			}	
		}
		for (int j = 0; j < J; ++j) {
			for (int itcol = 0; itcol < samples.cols[j].size(); ++itcol) {
				int i = samples.cols[j][itcol];
				termS_minus[j][k] += 0.5 * mA[i][k] - 2.0 * lambda(samples.Zcols[j][itcol])
									* ( mS[j][k] * vA[i][k] 
									+ mA[i][k] * dot(mA[i], mS[j]) 
									+ mb * mA[i][k] );
	
			}
		}
	}
	for (int k = 0; k < K; ++k) {
		for (int i = 0; i < I; ++i) {
			termA_minus[i][k] *= I / NSSrows;
			dCdA[i][k] = mA[i][k] / pvA[k] + termA_minus[i][k] + termA_plus[i][k];
			double d2CdA2 = 1.0 / vA[i][k];
			if (localSparsity && k == K - 2) {
				mA[i][k] = 1.0;
			} else {
				mA[i][k] = mA[i][k] - gamma * pow(d2CdA2, -alpha) * dCdA[i][k];
			}
		}	
		for (int j = 0; j < J; ++j) {
			termS_minus[j][k] *= J / NSScols;
			dCdS[j][k] = mS[j][k] / pvS[k] + termS_minus[j][k] + termS_plus[j][k];
			double d2CdS2 = 1.0 / vS[j][k];
			if (localSparsity && k == K - 1) {
				mS[j][k] = 1.0;
			} else {
				mS[j][k] = mS[j][k] - gamma * pow(d2CdS2, -alpha) * dCdS[j][k];
			}
		}
	}
}

void model::updateOffset() {
	double mb_num_plus = 0;
	double mb_den_plus = 0;
	double vb_plus = 0;
	for (int n = 0; n < data.X.size(); ++n) {
		int i = data.X[n][0];
		int j = data.X[n][1];
		vb_plus     -= 2.0 * lambda(data.Z[n]); 
		mb_num_plus += 0.5 + 2.0 * lambda(data.Z[n]) * dot(mA[i], mS[j]); 
		mb_den_plus -= 2.0 * lambda(data.Z[n]); 
	}
	double vb_minus = 0;
	double mb_num_minus = 0;
	double mb_den_minus = 0;
	for (int n = 0; n < samples.matrix.size(); ++n) {
		int i = samples.matrix[n][0];
		int j = samples.matrix[n][1];
		vb_minus     -= 2.0 * lambda(samples.Zmatrix[n]); 
		mb_num_minus += - 0.5 + 2 * lambda(samples.Zmatrix[n]) * dot(mA[i], mS[j]); 
		mb_den_minus -= 2 * lambda(samples.Zmatrix[n]); 
	}
	vb_minus     *= I * J / NSSmatrix;
	mb_num_minus *= I * J / NSSmatrix;
	mb_den_minus *= I * J / NSSmatrix;
	vb = 1.0 / (1.0 / pvb + vb_plus + vb_minus);
	mb = (mb_num_plus + mb_num_minus) / (1.0 / pvb + mb_den_plus + mb_den_minus);
}

void model::updatePriorVariances() {
	for (int k = 0; k < K; ++k) {
		double sum = 0;
		for (int i = 0; i < I; ++i) {
			sum	+= pow(mA[i][k], 2) + vA[i][k];
		}
		if (localSparsity && k == K - 2) {
			pvA[k] = 1;
		} else if (localSparsity && k == K - 1) {
			pvA[k] = 1e10;
		} else {
			pvA[k] = sum / I;
		}
	}
//	for (int k = 0; k < K; ++k) {
//		double sum = 0;
//		for (int j = 0; j < J; ++j) {
//			sum	+= pow(mS[j][k], 2) + vS[j][k];
//		}
//		if (localSparsity && k == K - 2) {
//			pvA[k] = 1e10;
//		} else if (localSparsity && k == K - 1) {
//			pvA[k] = 1;
//		} else {
//			pvS[k] = sum / J;
//		}
//	}
	//testGradPriorVariances(0);
}

void model::updateFittingParams() {
	double Zij;
	for(int n = 0; n < data.X.size(); ++n) {
		int i = data.X[n][0];
		int j = data.X[n][1];
		Zij = sqrt( commonTerm(i, j) );
		data.Z[n] = Zij;
	}
	for(int n = 0; n < samples.matrix.size(); ++n) {
		int i = samples.matrix[n][0];
		int j = samples.matrix[n][1];
		Zij = sqrt( commonTerm(i, j) );
		samples.Zmatrix[n] = Zij;
	}
	for(int i = 0; i < I; ++i) {
		for(int itrow = 0; itrow < samples.rows[i].size(); ++itrow) {
			int j = samples.rows[i][itrow];
			Zij = sqrt( commonTerm(i, j) );
			// TODO: FIX: In the next step the size of samples.cols can change!?
			samples.Zrows[i][itrow] = Zij;
		}
	}
	for(int j = 0; j < J; ++j) {
		for(int itcol = 0; itcol < NSScols; ++itcol) {
			int i = samples.cols[j][itcol];
			// Note order of j,i in Zcol indexing.
			Zij = sqrt( commonTerm(i, j) );
			samples.Zcols[j][itcol] = Zij;
		}
	}
}

double model::commonTerm(int i, int j) {
	// calculates:
	// vA(i,:)*mS(:,j)^2 + mA(:,i)^2*vS(:,j) + vA(i,:)*vS(:,j)
	// + (mA(i,:)*mS(:,j))^2 + 2*mb*mA(i,:)*mS(:,j) + mb^2 + vb
	double ans = dotSq(vA[i], mS[j]) + dotSq(vS[j], mA[i]) + dot(vA[i], vS[j])
				+ pow(dot(mA[i], mS[j]), 2) + 2.0 * mb * dot(mA[i], mS[j])
				+ mb * mb + vb;
	return ans;
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

double model::sigma(double z) {
	double ans = 1.0 / ( 1.0 + exp(-z) );
	return ans;
}

double model::lambda(double z) {
	double ans = (0.5 - sigma(z)) / (2.0 * z);
	return ans;
}
 
// N.b. can only test gradient when using all of the samples.
void model::testGradVariances() {
	double pert = 1e-6;
	int i = 0;
	int j = 0;
	int k = 0;
	vA[i][k] += pert;
	double Cplus = cost();
	vA[i][k] -= 2 * pert;
	double Cminus = cost();
	cout << "vA[" << i << "][" << k << "] = " << vA[i][k] + pert << endl;
	cout << "dCdvA[" << i << "][" << k << "] = " << (Cplus - Cminus) / (2 * pert) << endl;
	vA[i][k] += pert;
	vS[j][k] += pert;
	Cplus = cost();
	vS[j][k] -= 2 * pert;
	Cminus = cost();
	cout << "vS[" << j << "][" << k << "] = " << vS[j][k] + pert << endl;
	cout << "dCdvS[" << j << "][" << k << "] = " << (Cplus - Cminus) / (2 * pert) << endl;
	vS[j][k] += pert;
}

void model::testGradMeans(int i, int j, int k) {
	double pert = 1e-4;
	mA[i][k] += pert;
	double Cplus = cost();
	mA[i][k] -= 2 * pert;
	double Cminus = cost();
	cout << "mA[" << i << "][" << k << "] = " << mA[i][k] + pert << endl;
	cout << "dCdmA[" << i << "][" << k << "] = " << (Cplus - Cminus) / (2 * pert) << endl;
	mA[i][k] += pert;
	mS[j][k] += pert;
	Cplus = cost();
	mS[j][k] -= 2 * pert;
	Cminus = cost();
	mS[j][k] += pert;
	cout << "mS[" << j << "][" << k << "] = " << mS[j][k] + pert << endl;
	cout << "dCdmS[" << j << "][" << k << "] = " << (Cplus - Cminus) / (2 * pert) << endl;
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

double model::epoch(){
	// Sequence for perfoming optimisation
	updateFittingParams();
	updateVariances();
	updateMeans();
	updateOffset();
	if (updatePrior) { updatePriorVariances(); }
	return cost();
}

main(int argc, char *argv[]) {
	int I, J, K, NSSmatrix, NSSrows, NSScols;
	bool localSparsity, updatePrior;
	// Check number of CL arguments.
	if ( argc != 9 ) {
		cout << "Incorrect number of CL arguaments!\n";
	} else {
		string tmp = argv[1];
		I = atoi(tmp.c_str());
		tmp = argv[2];
		J = atoi(tmp.c_str());
		tmp = argv[3];
		K = atoi(tmp.c_str());
		tmp = argv[4];
		NSSmatrix = atoi(tmp.c_str());
		tmp = argv[5];
		NSScols = atoi(tmp.c_str()); // samples from cols, i.e. NSScols <= I
		tmp = argv[6];
		NSSrows = atoi(tmp.c_str());
		tmp = argv[7];
		updatePrior = (strcasecmp (tmp.c_str(), "true") == 0 || 
							atoi(tmp.c_str()) != 0);
		tmp = argv[8];
		localSparsity = (strcasecmp (tmp.c_str(), "true") == 0 || 
							atoi(tmp.c_str()) != 0);
	}

	model pa(I, J, K, NSSmatrix, NSSrows, NSScols, localSparsity, updatePrior);
	cout << "MODEL CREATED\n";
	pa.readData();
	cout << "DATA READ IN\n";
	pa.subSample();
	cout << "SUBSAMPLING PERFORMED\n";
	pa.initialiseReadIn();
	cout << "MODEL INITIALISED\n";
	
	// Find a sensible starting offset.
	pa.updateFittingParams();
	pa.updateOffset();
	double C = pa.cost();
	cout << "Initial cost = " << C << "\n";

	clock_t init, final;
	init = clock();	
	double Cold = 1e10;
	bool converged = false;
	int iter = 1;
	int timeStamp = 1;
	while ( iter < 3 || (!converged && iter < 250) ) {
		// Re-sample every so often.
		// Problem: may never converge as optimium keeps moving a little bit.
		if (iter % 1000 == 0) {
			pa.subSample();
			cout << "Resampled\n";
			Cold = 1e10;	
		}
		C = pa.epoch();
		if (iter % 10 == 0) {
			cout << "Iter: " << iter << ", cost = " << C << endl;
    		pa.C = C;
    		pa.Niters = iter;
			final = clock() - init;
  			pa.time = (double)final / (double)CLOCKS_PER_SEC;
			pa.saveData(timeStamp);
			++timeStamp;
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
    file.open("output/paSFSse_numTSs.txt");
    file << timeStamp;
    file.close();
	
	cout << "TOTAL TIME = " << pa.time << "s\n";

	return 0;
}
