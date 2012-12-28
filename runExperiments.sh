#!/bin/bash
# Command line arguments:
#  1. dataset number
#  2. number of complete repetitions
#  3. number of iterations
#  4. bool run dense experiments

if [ $# -ne 4 ]
then
  echo "Wrong # CL arguments."
  exit 1
fi

DATASET=$1
if [ "$DATASET" == "1" ]
then
  I=200
  J=200
fi
K=10
NSSmat=$((I * J / 50))
NSSI=$((I / 10))
NSSJ=$((J / 10))
DENSE=$4

rm -rf experiments_dataset_$1
mkdir experiments_dataset_$1
cd experiments_dataset_$1
NREPS=$2
for (( REP = 1; REP <= NREPS; REP++ ))  # Creates a new directory tree.
do
  echo $REP
  rm -rf experiment$REP
  mkdir experiment$REP
  cp -R ../code/* experiment$REP
  rm experiment$REP/results/*
  rm experiment$REP/data/*
  rm experiment$REP/Mcode/output/*
  rm experiment$REP/CCcode/output/G/*
  rm experiment$REP/CCcode/output/GnoPrior/*
  rm experiment$REP/CCcode/output/GnoPriorNoLocal/*
  rm experiment$REP/CCcode/output/S/*
  rm experiment$REP/CCcode/output/SnoPrior/*
  rm experiment$REP/CCcode/output/SnoPriorNoLocal/*
  rm experiment$REP/Rcode/SVI/noBiasedSampling/output/*
  rm experiment$REP/Rcode/SVI/biasedSampling/output/*
  cd experiment$REP/
  NITERS=$3
  for (( ITERNO = 1; ITERNO <= NITERS; ITERNO++ ))  # Overwrites all intermediate files.
  do
    cd Mcode/
    matlab -nosplash -nojvm -r "expSetup($DATASET, $ITERNO, $I, $J, $K), exit"
    matlab -nosplash -nojvm -r "expMatlabRoutines($K, $DENSE), exit"
    cd ../Rcode/SVI
    cd biasedSamplingNoPriorTuningNoLocalBias
    Rscript expRwrapper.R $I $J $K
    cd ../noBiasedSamplingNoPriorTuningNoLocalBias
    Rscript expRwrapper.R $I $J $K
    cd ../../../CCcode/
    # g++ -o VBMF_GF VBMF_GF.cc
    # chmod u+x VBMF_GF
    # last arg = update prior?
    ./VBMF_GF $I $J $K 0
    mkdir output/GnoPrior
    mv output/paGF* output/GnoPrior
    # g++ -o VBMF_SFSse VBMF_SFSse.cc
    # chmod u+x VBMF_SFSse
    # 2nd last arg = update prior? last = local offsets?
    ./VBMF_SFSse $I $J $K $NSSmat $NSSI $NSSJ 0 0
    mkdir output/SnoPriorNoLocal
    mv output/paS* output/SnoPriorNoLocal
    cd ../Mcode
    matlab -nosplash -nojvm -r "expAnalyseandSave($DENSE), exit"
    cd ../
  done
  cd ../
done
