##
# Load the data for the VBMF experiment.
# Run the R stochastic VB routines.
# Save the results for analysis.
#
# CL arguments: I J K

source("vmbm.R")

library(Matrix)


# Number of latent dimensions
tmp <- commandArgs(trailingOnly = TRUE)
I = as.integer( tmp[1] )
J = as.integer( tmp[2] )
K = as.integer( tmp[3] )

# Load data.
Xs <- as.matrix(read.table('../../../data/XSparse.txt', sep = ","))

# R is 1-indexed, matrix is 0-indexed for C++.
Xs <- Xs + 1
IfromMatrix  <- as.integer(max(Xs[ ,1]))
JfromMatrix  <- as.integer(max(Xs[ ,2]))
if (I != IfromMatrix) {
	print("I incorrect of matrix has trailing empty rows.")
	print(I)
	print(IfromMatrix)
}
if (J != JfromMatrix) {
	print("J incorrect of matrix has trailing empty columns.")
	print(J)
	print(JfromMatrix)
}

# Load initialisation.
mA <- as.matrix(read.table('../../../initialisation/mA_init.txt', sep = ","))
mS <- as.matrix(read.table('../../../initialisation/mS_init.txt', sep = ","))
mb <- as.double(read.table('../../../initialisation/mb_init.txt'))

start <- proc.time()
numTSs <- 5
for (i in 1 : numTSs) {
	if (i == 1) {
		pa <- SGDfast(I, J, K, Xs, mA, mS, mb)
	} else {
		pa <- SGDfast(I, J, K, Xs, mA, mS, mb, pa)
	}
	write.table(pa$mPostU, paste("output/mA_TS", i, ".txt", sep = ""), col.names = F, row.names = F, append = F)
	write.table(pa$mPostV, paste("output/mS_TS", i, ".txt", sep = ""), col.names = F, row.names = F, append = F)
	write.table(pa$mPostB, paste("output/mb_TS", i, ".txt", sep = ""), col.names = F, row.names = F, append = F)
	write.table(pa$vPostU, paste("output/vA_TS", i, ".txt", sep = ""), col.names = F, row.names = F, append = F)
	write.table(pa$vPostV, paste("output/vS_TS", i, ".txt", sep = ""), col.names = F, row.names = F, append = F)
	write.table(pa$vPostB, paste("output/vb_TS", i, ".txt", sep = ""), col.names = F, row.names = F, append = F)
	tmp <- pa$vPriorUh
	tmp[pa$k - 1] <- 1
	write.table(tmp, paste("output/pvA_TS", i, ".txt", sep = ""), col.names = F, row.names = F, append = F)
	tmp <- pa$vPriorVh
	tmp[pa$k] <- 1
	write.table(tmp, paste("output/pvS_TS", i, ".txt", sep = ""), col.names = F, row.names = F, append = F)
	write.table(1, paste("output/pvb_TS", i, ".txt", sep = ""), col.names = F, row.names = F, append = F)
	tmp <- as.double(proc.time() - start)
	time <- tmp[3]
	write.table(time, paste("output/time_TS", i, ".txt", sep = ""), col.names = F, row.names = F, append = F)
}
write.table(numTSs, "output/numTSs.txt", col.names = F, row.names = F, append = F)
