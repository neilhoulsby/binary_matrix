#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include <Rmath.h>
#include <R_ext/Rdynload.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Auxiliary function logistic */

double logistic(double x) {

	return 1 / (1 + exp(-x));
}

/* Auxiliary function lamda */

double lambda(double zeta) {

	return (0.5 - logistic(zeta)) / (2 * zeta);
}

/**
 * Function which updates the posterior approximation for the bias
 * when we randomly sample a positive entry.
 *
 * @param	mPostB		Pointer to the posterior mean of the
 * 				bias.
 * @param	vPostB		Pointer to the posterior variance of the
 * 				bias.
 * @param	vPriorB 	Pointer to the prior variance of the bias.
 * @param	mPriorB 	Pointer to the prior mean of the bias.
 * @param	mPred		The mean of the prediction
 * 				for the current positive entry.
 * @param	vPred		The variance of the prediction
 * 				for the current positive entry.
 * @param	rho		The learning rate.
 * @param	nOnesSampling	Pointer to the vector with the cumulative
 * 				number of one entries of each transaction.
 *
 */

void refineBiasPositive(double *mPostB, double *vPostB, double vPriorB,
	double mPriorB, double mPred, double vPred, int T, double rho,
	int *nOnesSampling) {

	double eta1New, eta2New, eta1Prior, eta2Prior, eta1Post, eta2Post,
		zeta;

	zeta = sqrt(mPred * mPred + vPred);

	mPred -= *mPostB;

	eta2Post = 1 / *vPostB;
	eta1Post = *mPostB / *vPostB;
	eta2Prior <- 1 / vPriorB;
	eta1Prior <- mPriorB / vPriorB;

	eta2New = eta2Prior - 4 * nOnesSampling[ T - 1 ] * lambda(zeta);

	eta1New = nOnesSampling[ T - 1 ] + 4 * lambda(zeta) *
		nOnesSampling[ T - 1 ] * mPred + eta1Prior;

	/* We update the posterior mean and variance */

	eta2Post = (1 - rho) * eta2Post + rho * eta2New;
	eta1Post = (1 - rho) * eta1Post + rho * eta1New;

	*vPostB = 1 / eta2Post;
	*mPostB = eta1Post / eta2Post;
}

/**
 * Function which updates the posterior approximation for the bias
 * when we randomly sample a negative entry.
 *
 * @param	mPostB		Pointer to the posterior mean of the
 * 				bias.
 * @param	vPostB		Pointer to the posterior variance of the
 * 				bias.
 * @param	vPriorB 	Pointer to the prior variance of the bias.
 * @param	mPriorB 	Pointer to the prior mean of the bias.
 * @param	mPred		The mean of the prediction
 * 				for the current positive entry.
 * @param	vPred		The variance of the prediction
 * 				for the current positive entry.
 * @param	rho		The learning rate.
 * @param	nZerosSampling	Pointer to the vector with the cumulative
 * 				number of zero entries of each transaction.
 * @param	e0		Pointer to the prior probability that a zero
 * 				entry is actually a one.
 *
 */

void refineBiasNegative(double *mPostB, double *vPostB, double vPriorB,
	double mPriorB, double mPred, double vPred, int T, double rho,
	int *nZerosSampling, double *e0) {

	double eta1New, eta2New, eta1Prior, eta2Prior, eta1Post, eta2Post,
		zeta, e;

	zeta = sqrt(mPred * mPred + vPred);
	e = logistic(mPred + log(*e0 / (1 - *e0)));

	mPred -= *mPostB;

	eta2Post = 1 / *vPostB;
	eta1Post = *mPostB / *vPostB;
	eta2Prior <- 1 / vPriorB;
	eta1Prior <- mPriorB / vPriorB;

	eta2New = eta2Prior - 4 * nZerosSampling[ T - 1 ] * lambda(zeta);

	eta1New = -(1 - 2 * e) * nZerosSampling[ T - 1 ] + 4 * lambda(zeta) *
		nZerosSampling[ T - 1 ] * mPred + eta1Prior;

	/* We update the posterior mean and variance */

	eta2Post = (1 - rho) * eta2Post + rho * eta2New;
	eta1Post = (1 - rho) * eta1Post + rho * eta1New;

	*vPostB = 1 / eta2Post;
	*mPostB = eta1Post / eta2Post;

}

/**
 * Function which updates the posterior approximation for the j-th row of V
 * when randomly sample a positive entry.
 *
 * @param	mPostU		Pointer to the posterior mean of the
 * 				entries of U.
 * @param	vPostU		Pointer to the posterior variance of the
 * 				entries of U.
 * @param	mPostV		Pointer to the posterior mean of the
 * 				entries of V.
 * @param	vPostV		Pointer to the posterior variance of the
 * 				entries of V.
 * @param	vPriorVj 	Pointer to the first component of the
 * 				prior variance	of the entries of U.
 * @param	vPriorVh 	Pointer to the second component of the
 * 				prior variance	of the entries of V.
 * @param	mPriorV 	Pointer to the prior mean of the entries of V.
 * @param	mPred		Pointer to the mean of the prediction
 * 				for the current positive entry.
 * @param	vPred		Pointer to the variance of the prediction
 * 				for the current positive entry.
 * @param 	i		The row number of the positive entry.
 * @param	j		The column number of the positive entry.
 * @param	T		The total number of transactions.
 * @param	P		The total number of products.
 * @param	k		The latent dimension.
 * @param	nOnesSampling	Pointer to the vector with the cumulative
 * 				number of one entries of each transaction.
 * @param	nZerosSampling	Pointer to the vector with the cumulative
 * 				number of zero entries of each transaction.
 * @param	nOnesAuxj	Pointer to the vector with the number of
 * 				ones for each row.
 * @param	nZerosAuxj	Pointer to the vector with the number of
 * 				zeros for each row.
 * @param	rho		The learning rate.
 *
 */

void refineRowVPositive(double *mPostU, double *vPostU, double *mPostV,
	double *vPostV, double *vPriorVj, double *vPriorVh, double *mPriorV,
	double *mPred, double *vPred, int i, int j, int T, int P,
	int k, int *nOnesSampling, int *nZerosSampling, int *nOnesAuxj,
	int *nZerosAuxj, double rho) {

	int h;
	double eta1New, eta2New, eta1Old, eta2Old, eta1Prior, eta2Prior,
		eta1Post, eta2Post, zeta, reps, vNew;

	/* We update the posterior approx. for the i-th row of U */

	for (h = 0 ; h < k ; h++) {

		zeta = sqrt((*mPred) * (*mPred) + *vPred);

		*mPred -= mPostU[ i + h * T ] * mPostV[ j + h * P ];
		*vPred -= mPostU[ i + h * T ] * mPostU[ i + h * T ] *
			vPostV[ j + h * P ] + vPostU[ i + h * T ] *
			mPostV[ j + h * P ] * mPostV[ j + h * P ] + 
			vPostU[ i + h * T ] * vPostV[ j + h * P ];

		eta2Prior = 1 / (vPriorVj[ j ] * vPriorVh[ h ]);
		eta1Prior = mPriorV[ h ] / (vPriorVj[ j ] * vPriorVh[ h ]);
		eta2Post = 1 / vPostV[ j + h * P ];
		eta1Post = mPostV[ j + h * P ] / vPostV[ j + h * P ];

		reps = (double) nZerosAuxj[ j ] * nOnesSampling[ T - 1 ] /
			nZerosSampling[ T - 1 ] + nOnesAuxj[ j ];

		eta2New = eta2Prior - 2 * reps * lambda(zeta) *
			(mPostU[ i + h * T ] * mPostU[ i + h * T ] +
			 vPostU[ i + h * T ]);

		eta1New = mPostU[ i + h * T ] * reps / 2 + 2 *
			lambda(zeta) * reps *
			mPostU[ i + h * T ] * (*mPred) + eta1Prior;

		/* We update the posterior mean and variance */

		eta2Post = (1 - rho) * eta2Post + rho * eta2New;
		eta1Post = (1 - rho) * eta1Post + rho * eta1New;

		vPostV[ j + h * P ] = 1 / eta2Post;
		mPostV[ j + h * P ] = eta1Post / eta2Post;

		/* We update the predictive mean and variance */

		*mPred += mPostU[ i + h * T ] * mPostV[ j + h * P ];
		*vPred += mPostU[ i + h * T ] * mPostU[ i + h * T ] *
			vPostV[ j + h * P ] + vPostU[ i + h * T ] *
			mPostV[ j + h * P ] * mPostV[ j + h * P ] + 
			vPostU[ i + h * T ] * vPostV[ j + h * P ];

		/* We update the prior variance */

		if (h != k - 1 && h != k - 2) {
			vNew = ((mPostV[ j + h * P ] - mPriorV[ h ]) *
				(mPostV[ j + h * P ] - mPriorV[ h ]) +
				vPostV[ j + h * P ]) / vPriorVj[ j ];
			vPriorVh[ h ] = (1 - rho) * vPriorVh[ h ] + rho * vNew;
		}
	}
}

/**
 * Function which updates the posterior approximation for the j-th row of V
 * when randomly sample a negative entry.
 *
 * @param	mPostU		Pointer to the posterior mean of the
 * 				entries of U.
 * @param	vPostU		Pointer to the posterior variance of the
 * 				entries of U.
 * @param	mPostV		Pointer to the posterior mean of the
 * 				entries of V.
 * @param	vPostV		Pointer to the posterior variance of the
 * 				entries of V.
 * @param	vPriorVj 	Pointer to the first component of the
 * 				prior variance	of the entries of U.
 * @param	vPriorVh 	Pointer to the second component of the
 * 				prior variance	of the entries of V.
 * @param	mPriorV 	Pointer to the prior mean of the entries of V.
 * @param	mPred		Pointer to the mean of the prediction
 * 				for the current positive entry.
 * @param	vPred		Pointer to the variance of the prediction
 * 				for the current positive entry.
 * @param 	i		The row number of the positive entry.
 * @param	j		The column number of the positive entry.
 * @param	T		The total number of transactions.
 * @param	P		The total number of products.
 * @param	k		The latent dimension.
 * @param	e0		Pointer to the prior probability that a
 * 				zero entry is actually a one entry.
 * @param	nOnesSampling	Pointer to the vector with the cumulative
 * 				number of one entries of each transaction.
 * @param	nZerosSampling	Pointer to the vector with the cumulative
 * 				number of zero entries of each transaction.
 * @param	nOnesAuxj	Pointer to the vector with the number of
 * 				ones for each row.
 * @param	nZerosAuxj	Pointer to the vector with the number of
 * 				zeros for each row.
 * @param	rho 		The learning rate.
 */

void refineRowVNegative(double *mPostU, double *vPostU, double *mPostV,
	double *vPostV, double *vPriorVj, double *vPriorVh, double *mPriorV,
	double *mPred, double *vPred, int i, int j, int T, int P,
	int k, double *e0, int *nOnesSampling, int *nZerosSampling,
	int *nOnesAuxj, int *nZerosAuxj, double rho) {

	int h;
	double eta1New, eta2New, eta1Old, eta2Old, eta1Prior, eta2Prior,
		eta1Post, eta2Post, zeta, reps, vNew, e;

	/* We update the posterior approx. for the i-th row of U */

	for (h = 0 ; h < k ; h++) {

		zeta = sqrt((*mPred) * (*mPred) + *vPred);
		e = logistic((*mPred) + log(*e0 / (1 - *e0)));

		*mPred -= mPostU[ i + h * T ] * mPostV[ j + h * P ];
		*vPred -= mPostU[ i + h * T ] * mPostU[ i + h * T ] *
			vPostV[ j + h * P ] + vPostU[ i + h * T ] *
			mPostV[ j + h * P ] * mPostV[ j + h * P ] + 
			vPostU[ i + h * T ] * vPostV[ j + h * P ];

		eta2Prior = 1 / (vPriorVj[ j ] * vPriorVh[ h ]);
		eta1Prior = mPriorV[ h ] / (vPriorVj[ j ] * vPriorVh[ h ]);
		eta2Post = 1 / vPostV[ j + h * P ];
		eta1Post = mPostV[ j + h * P ] / vPostV[ j + h * P ];

		reps = nZerosAuxj[ j ] + (double) nOnesAuxj[ j ] *
			nZerosSampling[ T - 1 ] / nOnesSampling[ T - 1 ];

		eta2New = eta2Prior - 2 * reps * lambda(zeta) *
			(mPostU[ i + h * T ] * mPostU[ i + h * T ] +
			 vPostU[ i + h * T ]);

		eta1New = -(1 - 2 * e) * mPostU[ i + h * T ] *
			reps / 2 + 2 * lambda(zeta) * reps *
			mPostU[ i + h * T ] * (*mPred) + eta1Prior;

		/* We update the posterior mean and variance */

		eta2Post = (1 - rho) * eta2Post + rho * eta2New;
		eta1Post = (1 - rho) * eta1Post + rho * eta1New;

		vPostV[ j + h * P ] = 1 / eta2Post;
		mPostV[ j + h * P ] = eta1Post / eta2Post;

		/* We update the predictive mean and variance */

		*mPred += mPostU[ i + h * T ] * mPostV[ j + h * P ];
		*vPred += mPostU[ i + h * T ] * mPostU[ i + h * T ] *
			vPostV[ j + h * P ] + vPostU[ i + h * T ] *
			mPostV[ j + h * P ] * mPostV[ j + h * P ] + 
			vPostU[ i + h * T ] * vPostV[ j + h * P ];

		/* We update the prior variance */

		if (h != k - 1 && h != k - 2) {
			vNew = ((mPostV[ j + h * P ] - mPriorV[ h ]) *
				(mPostV[ j + h * P ] - mPriorV[ h ]) +
				vPostV[ j + h * P ]) / vPriorVj[ j ];
			vPriorVh[ h ] = (1 - rho) * vPriorVh[ h ] + rho * vNew;
		}
	}
}

/**
 * Function which updates the posterior approximation for the i-th row of U
 * when randomly sample a positive entry.
 *
 * @param	mPostU		Pointer to the posterior mean of the
 * 				entries of U.
 * @param	vPostU		Pointer to the posterior variance of the
 * 				entries of U.
 * @param	mPostV		Pointer to the posterior mean of the
 * 				entries of V.
 * @param	vPostV		Pointer to the posterior variance of the
 * 				entries of V.
 * @param	vPriorUi 	Pointer to the first component of the
 * 				prior variance	of the entries of U.
 * @param	vPriorUh 	Pointer to the second component of the
 * 				prior variance	of the entries of U.
 * @param	mPriorU 	Pointer to the prior mean of the entries of U.
 * @param	mPred		Pointer to the mean of the prediction
 * 				for the current positive entry.
 * @param	vPred		Pointer to the variance of the prediction
 * 				for the current positive entry.
 * @param 	i		The row number of the positive entry.
 * @param	j		The column number of the positive entry.
 * @param	T		The total number of transactions.
 * @param	P		The total number of products.
 * @param	k		The latent dimension.
 * @param	nOnesSampling	Pointer to the vector with the cumulative
 * 				number of one entries of each transaction.
 * @param	nZerosSampling	Pointer to the vector with the cumulative
 * 				number of zero entries of each transaction.
 * @param	nOnesAuxi	Pointer to the vector with the number of
 * 				ones for each row.
 * @param	nZerosAuxi	Pointer to the vector with the number of
 * 				zeros for each row.
 * @param	rho		The learning rate.
 *
 */

void refineRowUPositive(double *mPostU, double *vPostU, double *mPostV,
	double *vPostV, double *vPriorUi, double *vPriorUh, double *mPriorU,
	double *mPred, double *vPred, int i, int j, int T, int P,
	int k, int *nOnesSampling, int *nZerosSampling, int *nOnesAuxi,
	int *nZerosAuxi, double rho) {

	int h;
	double eta1New, eta2New, eta1Old, eta2Old, eta1Prior, eta2Prior,
		eta1Post, eta2Post, zeta, reps, vNew;

	/* We update the posterior approx. for the i-th row of U */

	for (h = 0 ; h < k ; h++) {

		zeta = sqrt((*mPred) * (*mPred) + *vPred);

		*mPred -= mPostU[ i + h * T ] * mPostV[ j + h * P ];
		*vPred -= mPostU[ i + h * T ] * mPostU[ i + h * T ] *
			vPostV[ j + h * P ] + vPostU[ i + h * T ] *
			mPostV[ j + h * P ] * mPostV[ j + h * P ] + 
			vPostU[ i + h * T ] * vPostV[ j + h * P ];

		eta2Prior = 1 / (vPriorUi[ i ] * vPriorUh[ h ]);
		eta1Prior = mPriorU[ h ] / (vPriorUi[ i ] * vPriorUh[ h ]);
		eta2Post = 1 / vPostU[ i + h * T ];
		eta1Post = mPostU[ i + h * T ] / vPostU[ i + h * T ];

		reps = (double) nZerosAuxi[ i ] * nOnesSampling[ T - 1 ] /
			nZerosSampling[ T - 1 ] + nOnesAuxi[ i ];

		eta2New = eta2Prior - 2 * reps * lambda(zeta) *
			(mPostV[ j + h * P ] * mPostV[ j + h * P ] +
			 vPostV[ j + h * P ]);

		eta1New = mPostV[ j + h * P ] * reps / 2 + 2 *
			lambda(zeta) * reps * mPostV[ j + h * P ] *
			(*mPred) + eta1Prior;

		/* We update the posterior mean and variance */

		eta2Post = (1 - rho) * eta2Post + rho * eta2New;
		eta1Post = (1 - rho) * eta1Post + rho * eta1New;

		vPostU[ i + h * T ] = 1 / eta2Post;
		mPostU[ i + h * T ] = eta1Post / eta2Post;

		/* We update the predictive mean and variance */

		*mPred += mPostU[ i + h * T ] * mPostV[ j + h * P ];
		*vPred += mPostU[ i + h * T ] * mPostU[ i + h * T ] *
			vPostV[ j + h * P ] + vPostU[ i + h * T ] *
			mPostV[ j + h * P ] * mPostV[ j + h * P ] + 
			vPostU[ i + h * T ] * vPostV[ j + h * P ];
	}
}

/**
 * Function which updates the posterior approximation for the i-th row of U
 * when randomly sample a negative entry.
 *
 * @param	mPostU		Pointer to the posterior mean of the
 * 				entries of U.
 * @param	vPostU		Pointer to the posterior variance of the
 * 				entries of U.
 * @param	mPostV		Pointer to the posterior mean of the
 * 				entries of V.
 * @param	vPostV		Pointer to the posterior variance of the
 * 				entries of V.
 * @param	vPriorUi 	Pointer to the first component of the
 * 				prior variance	of the entries of U.
 * @param	vPriorUh 	Pointer to the second component of the
 * 				prior variance	of the entries of U.
 * @param	mPriorU 	Pointer to the prior mean of the entries of U.
 * @param	mPred		Pointer to the mean of the prediction
 * 				for the current positive entry.
 * @param	vPred		Pointer to the variance of the prediction
 * 				for the current positive entry.
 * @param 	i		The row number of the positive entry.
 * @param	j		The column number of the positive entry.
 * @param	T		The total number of transactions.
 * @param	P		The total number of products.
 * @param	k		The latent dimension.
 * @param	e0		Pointer to the prior probability that a zero
 * 				entry is actualy a one entry.
 * @param	nOnesSampling	Pointer to the vector with the cumulative
 * 				number of one entries of each transaction.
 * @param	nZerosSampling	Pointer to the vector with the cumulative
 * 				number of zero entries of each transaction.
 * @param	nOnesAuxi	Pointer to the vector with the number of
 * 				ones for each row.
 * @param	nZerosAuxi	Pointer to the vector with the number of
 * 				zeros for each row.
 * @param	rho 		The learning rate.
 *
 */

void refineRowUNegative(double *mPostU, double *vPostU, double *mPostV,
	double *vPostV, double *vPriorUi, double *vPriorUh, double *mPriorU,
	double *mPred, double *vPred, int i, int j, int T, int P,
	int k, double *e0, int *nOnesSampling, int *nZerosSampling,
	int *nOnesAuxi, int *nZerosAuxi, double rho) {

	int h;
	double eta1New, eta2New, eta1Old, eta2Old, eta1Prior, eta2Prior,
		eta1Post, eta2Post, zeta, reps, vNew, e;

	/* We update the posterior approx. for the i-th row of U */

	for (h = 0 ; h < k ; h++) {

		zeta = sqrt((*mPred) * (*mPred) + *vPred);
		e = logistic((*mPred) + log(*e0 / (1 - *e0)));

		*mPred -= mPostU[ i + h * T ] * mPostV[ j + h * P ];
		*vPred -= mPostU[ i + h * T ] * mPostU[ i + h * T ] *
			vPostV[ j + h * P ] + vPostU[ i + h * T ] *
			mPostV[ j + h * P ] * mPostV[ j + h * P ] + 
			vPostU[ i + h * T ] * vPostV[ j + h * P ];

		eta2Prior = 1 / (vPriorUi[ i ] * vPriorUh[ h ]);
		eta1Prior = mPriorU[ h ] / (vPriorUi[ i ] * vPriorUh[ h ]);
		eta2Post = 1 / vPostU[ i + h * T ];
		eta1Post = mPostU[ i + h * T ] / vPostU[ i + h * T ];

		reps = nZerosAuxi[ i ] + nOnesAuxi[ i ] *
			(double) nZerosSampling[ T - 1 ] /
			nOnesSampling[ T - 1 ];

		eta2New = eta2Prior - 2 * reps * lambda(zeta) *
			(mPostV[ j + h * P ] * mPostV[ j + h * P ] +
			 vPostV[ j + h * P ]);

		eta1New = -(1 - 2 * e) * mPostV[ j + h * P ] * reps / 2 + 2 *
			lambda(zeta) * reps * mPostV[ j + h * P ] *
			(*mPred) + eta1Prior;

		/* We update the posterior mean and variance */

		eta2Post = (1 - rho) * eta2Post + rho * eta2New;
		eta1Post = (1 - rho) * eta1Post + rho * eta1New;

		vPostU[ i + h * T ] = 1 / eta2Post;
		mPostU[ i + h * T ] = eta1Post / eta2Post;

		/* We update the predictive mean and variance */

		*mPred += mPostU[ i + h * T ] * mPostV[ j + h * P ];
		*vPred += mPostU[ i + h * T ] * mPostU[ i + h * T ] *
			vPostV[ j + h * P ] + vPostU[ i + h * T ] *
			mPostV[ j + h * P ] * mPostV[ j + h * P ] + 
			vPostU[ i + h * T ] * vPostV[ j + h * P ];
	}
}

/* Auxiliary function to get the list element named str, or return NULL */

SEXP getListElement(SEXP list, const char *str) {

	SEXP elmt = R_NilValue, names = getAttrib(list, R_NamesSymbol);
	int i;

	for (i = 0; i < length(list); i++)
		if(strcmp(CHAR(STRING_ELT(names, i)), str) == 0) {
			elmt = VECTOR_ELT(list, i);
			break;
		}
	
	return elmt;
}

/**
 * Function which selects a transaction randomly according to some weights.
 *
 * @param T		The total number of transactions.
 * @param weights	The cumulative weight given to each transaction
 * 			(integer vector).
 */

int sampleTransaction(int T, int *weights) {

	int randomWeight;
	int imax = T - 1;
	int imin = 0;
	int imid;

	/* We select a weight value randomly */

	randomWeight = ((int) (unif_rand() * (weights[ T - 1 ]))) + 1;

	/* We identify the first transaction whose cumulative
	 * weight is larger or equal to randomWeight */

	/* We look for the element using binary search */

	while (imax >= imin) {

		imid = imin + ((imax - imin) / 2);

		if (weights[ imid ] < randomWeight)
			imin = imid + 1;
		else if (weights[ imid ] > randomWeight)
			imax = imid - 1;
		else {
			while (imid > 0 && weights[ imid - 1 ] == randomWeight)
				imid--;

			return imid;
		}
	}

	if (weights[ imid ] > randomWeight)
		return imid;

	return imid + 1;
}

/**
 * Function which determines whether an element is in a sorted integer array.
 *
 * @param elem	Elemnt to look for in the array.
 * @param p	Pointer to the begining of the array.
 * @param size	Size of the array.
 *
 * @return	1 if the element is found and 0 otherwise.
 *
 */

int isElementInSortedArray(int elem, int *p, int size) {

	int imax = size - 1;
	int imin = 0;
	int imid;

	/* We look for the element using binary search */

	while (imax >= imin) {

		imid = imin + ((imax - imin) / 2);

		if (p[ imid ] < elem)
			imin = imid + 1;
		else if (p[ imid ] > elem)
			imax = imid - 1;
		else
			return 1;
	}

	return 0;
}

/**
 * Function which samples the row and column of a negative entry
 *
 * @param jM			Integer pointer to the field "j" of the sparse
 * 				transaction matrix (in compressed row format).
 * @param pM			Integer pointer to the field "p" of the sparse
 * 				transaction matrix (in compressed row format).
 * @param T			The total number of transactions.
 * @param P			The total number of products.
 * @param nZerosSampling	Vector with the cumulative number of zeros for
 * 				each transaction.
 * @param i			Pointer to the place where the row of the
 * 				positive entry will be stored.
 * @param j			Pointer to the place where the column of the
 * 				positive entry will be stored.
 */

void sampleNegativeEntry(int *jM, int *pM, int T, int P, int *nZerosSampling,
	int *i, int *j) {

	/* We select a transaction randomly */

	*i = sampleTransaction(T, nZerosSampling);

	/* We select a product randomly */

	do { 

		*j = (int) (unif_rand() * P);

	} while (isElementInSortedArray(*j, jM + pM[ *i ], pM[ *i + 1 ] - pM[ *i ]));
}

/**
 * Function which samples the row and column of a positive entry
 *
 * @param jM		Integer pointer to the field "j" of the sparse
 * 			transaction matrix (in compressed row format).
 * @param pM		Integer pointer to the field "p" of the sparse
 * 			transaction matrix (in compressed row format).
 * @param T		The total number of transactions.
 * @param nOnes		Vector with the cumulative number of ones for each
 * 			transaction.
 * @param i		Pointer to the place where the row of the positive
 * 			entry will be stored.
 * @param j		Pointer to the place where the column of the positive
 * 			entry will be stored.
 */

void samplePositiveEntry(int *jM, int *pM, int T, int *nOnes, int *i, int *j) {

	/* We select a transaction randomly */

	*i = sampleTransaction(T, nOnes);

	/* Select randomly a product */

	*j = jM[ (int) (unif_rand() * (pM[ *i + 1 ] - pM[ *i ])) + pM[ *i ] ];
}

/**
 * Function that fits a Bayesian linear classifier to each product.
 *
 * @param R_T		R object with the number of transactions in the
 * 			training set (integer).
 * @param R_P		R object with the number of products (P) in the
 * 			training set (integer).
 * @param R_j		R object with the field "j" of the sparse
 * 			transaction matrix (in compressed row format).
 * @param R_p		R object with the field "p" of the sparse transaction
 * 			matrix (in compressed row format).
 * @param R_nIter	R object with the number of iterations of
 * 			stochastic VI.
 * @param R_m	R object with a list containing the parameters of the Bayesian
 * 		model
 *
 * @return	
 *
 */

SEXP svi(SEXP R_T, SEXP R_P, SEXP R_j, SEXP R_p, SEXP R_nIter, SEXP R_m) {

	int T, P, nIter, k, h;
	int *jM, *pM, iter, i, j;

	double *mPostU, *vPostU, *mPostV, *vPostV, *mPostB, *vPostB, *mPriorU,
		*mPriorV, *vPriorUi, *vPriorUh, *vPriorVj, *vPriorVh,
		vPriorB, mPriorB, mPred, vPred, e, *e0, rho;

	int *nZerosAuxi, *nOnesAuxi, *nZerosAuxj, *nOnesAuxj, *nOnesSampling,
		*nZerosSampling;
	
	/* We read in the random number generator seed */

	GetRNGstate();

        /* We map the R variables to c variables */

        jM = INTEGER_POINTER(R_j);
        pM = INTEGER_POINTER(R_p);
        T = *INTEGER_POINTER(R_T);
        P = *INTEGER_POINTER(R_P);
        nIter = *INTEGER_POINTER(R_nIter);

	mPostU = NUMERIC_POINTER(getListElement(R_m, "mPostU"));
	vPostU = NUMERIC_POINTER(getListElement(R_m, "vPostU"));
	mPostV = NUMERIC_POINTER(getListElement(R_m, "mPostV"));
	vPostV = NUMERIC_POINTER(getListElement(R_m, "vPostV"));
	mPostB = NUMERIC_POINTER(getListElement(R_m, "mPostB"));
	vPostB = NUMERIC_POINTER(getListElement(R_m, "vPostB"));
	mPriorU = NUMERIC_POINTER(getListElement(R_m, "mPriorU"));
	mPriorV = NUMERIC_POINTER(getListElement(R_m, "mPriorV"));
	vPriorUi = NUMERIC_POINTER(getListElement(R_m, "vPriorUi"));
	vPriorUh = NUMERIC_POINTER(getListElement(R_m, "vPriorUh"));
	mPriorU = NUMERIC_POINTER(getListElement(R_m, "mPriorU"));
	vPriorVj = NUMERIC_POINTER(getListElement(R_m, "vPriorVj"));
	vPriorVh = NUMERIC_POINTER(getListElement(R_m, "vPriorVh"));
	mPriorB = *NUMERIC_POINTER(getListElement(R_m, "mPriorB"));
	vPriorB = *NUMERIC_POINTER(getListElement(R_m, "vPriorB"));
	e0 = NUMERIC_POINTER(getListElement(R_m, "e0"));
	k = *INTEGER_POINTER(getListElement(R_m, "k"));
	nOnesAuxi = INTEGER_POINTER(getListElement(R_m, "nOnesAuxi"));
	nOnesAuxj = INTEGER_POINTER(getListElement(R_m, "nOnesAuxj"));
	nZerosAuxi = INTEGER_POINTER(getListElement(R_m, "nZerosAuxi"));
	nZerosAuxj = INTEGER_POINTER(getListElement(R_m, "nZerosAuxj"));
        nOnesSampling = INTEGER_POINTER(getListElement(R_m, "nOnesSampling"));
        nZerosSampling =
		INTEGER_POINTER(getListElement(R_m, "nZerosSampling"));

	/* We start the stochastic optimization */

	for (iter = 1 ; iter <= nIter ; iter++) {

		rho = 0.01;

		if (unif_rand() <= 0.5) {

			/* We sample the row and column of a positive entry */

			samplePositiveEntry(jM, pM, T, nOnesSampling, &i, &j);

			/* We compute the predictive mean and variance */

			mPred = 0;
			vPred = 0;
			for (h = 0 ; h < k ; h++) {
				mPred += mPostU[ i + h * T ] *
					mPostV[ j + h * P ];
				vPred += mPostU[ i + h * T ] *
					mPostU[ i + h * T ] *
					vPostV[ j + h * P ] +
					vPostU[ i + h * T ] *
					mPostV[ j + h * P ] *
					mPostV[ j + h * P ] + 
					vPostU[ i + h * T ] *
					vPostV[ j + h * P ];
			}
			mPred += *mPostB;
			vPred += *vPostB;

			/* We refine the parameters for the i-th row of U */

			refineRowUPositive(mPostU, vPostU, mPostV, vPostV,
				vPriorUi, vPriorUh, mPriorU, &mPred, &vPred,
				i, j, T, P, k, nOnesSampling, nZerosSampling,
				nOnesAuxi, nZerosAuxi, rho);

			/* We refine the parameters for the i-th row of V */

			refineRowVPositive(mPostU, vPostU, mPostV, vPostV,
				vPriorVj, vPriorVh, mPriorV, &mPred, &vPred,
				i, j, T, P, k, nOnesSampling, nZerosSampling,
				nOnesAuxj, nZerosAuxj, rho);

			/* We refine the global bias parameter */

			refineBiasPositive(mPostB, vPostB, vPriorB, mPriorB,
				mPred, vPred, T, rho, nOnesSampling);

		} else {

			/* We sample the row and column of a negative entry */

			sampleNegativeEntry(jM, pM, T, P, nZerosSampling,
				&i, &j);

			/* We compute the predictive mean and variance */

			mPred = 0;
			vPred = 0;
			for (h = 0 ; h < k ; h++) {
				mPred += mPostU[ i + h * T ] *
					mPostV[ j + h * P ];
				vPred += mPostU[ i + h * T ] *
					mPostU[ i + h * T ] *
					vPostV[ j + h * P ] +
					vPostU[ i + h * T ] *
					mPostV[ j + h * P ] *
					mPostV[ j + h * P ] + 
					vPostU[ i + h * T ] *
					vPostV[ j + h * P ];
			}
			mPred += *mPostB;
			vPred += *vPostB;

			/* We refine the parameters for the i-th row of U */

			refineRowUNegative(mPostU, vPostU, mPostV, vPostV,
				vPriorUi, vPriorUh, mPriorU, &mPred, &vPred,
				i, j, T, P, k, e0, nOnesSampling,
				nZerosSampling, nOnesAuxi, nZerosAuxi, rho);

			/* We refine the parameters for the i-th row of V */

			refineRowVNegative(mPostU, vPostU, mPostV, vPostV,
				vPriorVj, vPriorVh, mPriorV, &mPred, &vPred,
				i, j, T, P, k, e0, nOnesSampling,
				nZerosSampling, nOnesAuxj, nZerosAuxj, rho);

			/* We refine the global bias parameter */

			refineBiasNegative(mPostB, vPostB, vPriorB, mPriorB,
				mPred, vPred, T, rho, nZerosSampling, e0);
		}

		if (iter % 100000 == 0) {
			fprintf(stdout, "%d\n", iter);
			fflush(stdout);
		}
	}

	/* We write out the random number generator seed */

	PutRNGstate();

	/* We free memory */

	return R_m;
}
