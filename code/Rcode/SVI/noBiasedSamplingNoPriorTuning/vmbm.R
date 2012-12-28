
library(Matrix)

dyn.load("vmbm.so")

##
# The lambda function
#

lambda <- function(x) (0.5 - logistic(x)) / (2 * x)

##
# The logistic function
#

logistic <- function(x) 1 / (1 + exp(-x))

##
# Function which optimizes the variational lower bound by
# stochastic gradient descent.
#
# @param n	The number of rows in the data matrix.
# @param d	The number of columns in the data matrix.
# @param k 	The latent dimension.
# @param ones	An nOnes x 2 matrix with the coordinates of
#		the ones in the data matrix.
# @param mPostU Optional parameter with the initial value for mPostU.
# @param mPostV Optional parameter with the initial value for mPostV.
# @param mPostB Optional parameter with the initial value for mPostB.
# @param m	Optional parameter which gives the initialization for
#		the stochastic optimization method.
# 
# @return	A list with the optimized model. The important elements
#		in this list are:
#
#		mPostU	-> The posterior means for U.
#		vPostU	-> The posterior variances for U.
#		mPostV	-> The posterior means for V.
#		vPostV	-> The posterior variances for V.
#		mPostB	-> The posterior mean for the bias.
#		vPostB	-> The posterior variance for the bias.
#		mPriorUh -> The prior variances for the columns of U.
#		mPriorVh -> The prior variances for the columns of V.
#

SGDfast <- function(n, d, k, ones, mPostU = NULL, mPostV = NULL, mPostB = NULL, m = NULL) {

	# We create a sparse matrix with the data

	mbd <- sparseMatrix(i = ones[ , 1 ], j = ones[ ,2 ], dims = c(n, d))
	mbd <- as(mbd, "RsparseMatrix")

	# We increase the latent dimension to incorporate the bias parameters

	if (is.null(m)) {

		k <- as.integer(k + 2)

		m <- list()
		m$n <- n
		m$k <- k
		m$d <- d

		# We initialize the prior matrices

		m$vPriorUi <- rep(1, n)
		m$vPriorUh <- rep(1, k)
		m$mPriorU <- rep(0, k)
		m$mPriorU[ k - 1 ] <- 1
		m$vPriorUh[ k - 1 ] <- 1e-10
		m$vPriorUh[ k ] <- 1e10

		m$vPriorVj <- rep(1, d)
		m$vPriorVh <- rep(1, k)
		m$mPriorV <- rep(0, k)
		m$mPriorV[ k ] <- 1
		m$vPriorVh[ k ] <- 1e-10
		m$vPriorVh[ k - 1 ] <- 1e10

		m$vPriorB <- 1e10
		m$mPriorB <- 0

		# We initialize the posterior means

		m$mPostU <- matrix(rnorm(n * k), n, k)
		if (!is.null(mPostU))
			m$mPostU[ , 1 : (k - 2) ] <- mPostU
		m$mPostU[ , k - 1 ] <- 1
		m$mPostU[ , k ] <- 0

		m$mPostV <- matrix(rnorm(d * k), d, k)
		if (!is.null(mPostV))
			m$mPostV[ , 1 : (k - 2) ] <- mPostV
		m$mPostV[ , k - 1 ] <- 0
		m$mPostV[ , k ] <- 1

		m$mPostB <- 0
		if (!is.null(mPostB))
			m$mPostB <- mPostB

		# We initialize the posterior variances

		m$vPostU <- matrix(1, n, k)
		m$vPostU[ , k - 1 ] <- 1e-10

		m$vPostV <- matrix(1, d, k)
		m$vPostV[ , k ] <- 1e-10

		m$vPostB <- 1
		m$e0 <- 0

		m$nOnesAuxi <- as.integer(as.double(mbd %*% rep(1, d)))
		m$nZerosAuxi <- d - m$nOnesAuxi
		m$nOnesAuxj <- as.integer(as.double(rep(1, n) %*% mbd))
		m$nZerosAuxj <- n - m$nOnesAuxj
		m$nOnesSampling <- cumsum(m$nOnesAuxi)
		m$nZerosSampling <- cumsum(m$nZerosAuxi)
	}

	# We start the stochastic gradient descent optimization

	m <- .Call("svi", n, d, mbd@j, mbd@p, as.integer(1e6), m, dup = FALSE)

	m
}
