#!/bin/bash
# The config setup and usage was being improved throughout the project,
# so the usage is not consistent everywhere,
# hence this file with all the commands in the
# logical order corresponding to the figures.

# Fig.2
for task in toyplain toydeceptive; do
  for algo in pbt pb2rand bgpbt firepbt; do
    for seed in {0..4}; do
      python run.py --config-name ${task}_${algo}_00 server=star04 ++general.seed_offset=${seed};
      done; done; done;

# Fig.3 and Fig.4 show results that are subsets of the results in Fig.5

# Fig.5 (a,b)
for task in fmnist c10; do
  for algo in pbt pb2rand bgpbt firepbt; do
    for config in {15..18}; do
      for seed in {0..4}; do
        python run.py --config-name ${task}_${algo}_${config} server=star04 ++general.seed_offset=${seed};
        done; done; done; done;

# Fig.5 (c)
configs=(14 12 10 13);
suffixes=(step50e6 step15e6 step5e6 step15e5);
for task in hopper; do
  for algo in pbt pb2rand pb2mix bgpbt firepbt; do
    for i in "${!configs[@]}"; do
      config=${configs[$i]};
      suffix=${suffixes[$i]};
      for seed in {0..6}; do
        python run.py --config-name ${task}_${config} server=star04 algo=${algo}_${suffix} ++general.seed_offset=${seed};
        done; done; done; done;

# Fig.5 (d)
configs=(17 14 13 12);
suffixes=(step50e6 step15e6 step5e6 step15e5);
for task in humanoid; do
  for algo in pbt pb2rand pb2mix bgpbt firepbt; do
    for i in "${!configs[@]}"; do
      config=${configs[$i]};
      suffix=${suffixes[$i]};
      for seed in {0..6}; do
        python run.py --config-name ${task}_${config} server=star04 algo=${algo}_${suffix} ++general.seed_offset=${seed};
        done; done; done; done;

# Fig.6 (a, b). Small search space already ready from fig. 5
for task in fmnist; do
  for algo in pbt pb2rand pb2mix bgpbt firepbt; do
    for config in 19; do
      for seed in {0..4}; do
        python run.py --config-name ${task}_${algo}_${config} server=star04 ++general.seed_offset=${seed};
        done; done; done; done;
for task in c10; do
  for algo in pbt pb2rand pb2mix bgpbt firepbt; do
    for config in 19; do
      for seed in {0..4}; do
        python run.py --config-name ${task}_${config} server=star04 algo=${algo} ++general.seed_offset=${seed};
        done; done; done; done;
for task in fmnist c10; do
  for algo in pbt pb2rand pb2mix bgpbt firepbt; do
    for config in 20; do
      for seed in {0..4}; do
        python run.py --config-name ${task}_${config} server=star04 algo=${algo} ++general.seed_offset=${seed};
        done; done; done; done;

# Fig.6 (c). Large search space already ready from fig. 5
configs=(11 17);
suffixes=(step5e6 step5e6); # same
for task in hopper; do
  for algo in pbt pb2rand bgpbt firepbt; do
    for i in "${!configs[@]}"; do
      config=${configs[$i]};
      suffix=${suffixes[$i]};
      for seed in {0..6}; do
        python run.py --config-name ${task}_${config} server=star04 algo=${algo}_${suffix} ++general.seed_offset=${seed};
        done; done; done; done;
for seed in {0..6}; do
  python run.py --config-name hopper_17 server=star04 algo=pb2mix_step5e6 ++general.seed_offset=${seed};
  done;

# Fig.6 (d). Large search space already ready from fig. 5
configs=(22 23);
suffixes=(step5e6 step5e6); # same
for task in humanoid; do
  for algo in pbt pb2rand bgpbt firepbt; do
    for i in "${!configs[@]}"; do
      config=${configs[$i]};
      suffix=${suffixes[$i]};
      for seed in {0..6}; do
        python run.py --config-name ${task}_${config} server=star04 algo=${algo}_${suffix} ++general.seed_offset=${seed};
        done; done; done; done;
for seed in {0..6}; do
  python run.py --config-name humanoid_23 server=star04 algo=pb2mix_step5e6 ++general.seed_offset=${seed};
  done;

# Fig. 7 (a,b)
configs=(22 24);
suffixes=(pop8 pop50);
for task in fmnist c10; do
  for algo in pbt pb2rand pb2mix bgpbt firepbt; do
    for i in "${!configs[@]}"; do
      config=${configs[$i]};
      suffix=${suffixes[$i]};
      for seed in {0..4}; do
        python run.py --config-name ${task}_${config} server=star04 algo=${algo}_${suffix} ++general.seed_offset=${seed};
        done; done; done; done;

# Fig.7 (c)
configs=(15 16);
suffixes=(pop8_step5e6 pop50_step5e6);
for task in hopper; do
  for algo in pbt pb2rand pb2mix bgpbt firepbt; do
    for i in "${!configs[@]}"; do
      config=${configs[$i]};
      suffix=${suffixes[$i]};
      for seed in {0..6}; do
        python run.py --config-name ${task}_${config} server=star04 algo=${algo}_${suffix} ++general.seed_offset=${seed};
        done; done; done; done;

# Fig.7 (d)
configs=(18 19);
suffixes=(pop8_step5e6 pop50_step5e6);
for task in humanoid; do
  for algo in pbt pb2rand pb2mix bgpbt firepbt; do
    for i in "${!configs[@]}"; do
      config=${configs[$i]};
      suffix=${suffixes[$i]};
      for seed in {0..6}; do
        python run.py --config-name ${task}_${config} server=star04 algo=${algo}_${suffix} ++general.seed_offset=${seed};
        done; done; done; done;

# Fig. 8 reuses data from previous experiments

# Fig.9 (a)
for algo in pbt pb2rand pb2mix bgpbt firepbt; do
  for seed in {0..4}; do
    python run.py --config-name c100_01 server=star04 algo=${algo} ++general.seed_offset=${seed};
    done; done;

# Fig.9 (b)
for algo in pbt pb2rand pb2mix bgpbt firepbt; do
  for seed in {0..4}; do
    python run.py --config-name timg_12 server=star04 algo=${algo}_step1500_tmax150k ++general.seed_offset=${seed};
    done; done;

# Fig.9 (c, d)
for task in pusher walker; do
  for algo in pbt pb2rand pb2mix bgpbt firepbt; do
    for seed in {0..6}; do
      python run.py --config-name ${task}_01 server=star04 algo=${algo}_step5e6 ++general.seed_offset=${seed};
      done; done; done;

# Fig. 10 reuses data from previous experiments

# Fig.11
for algo in pbt pb2rand pb2mix bgpbt firepbt; do
  for seed in {0..4}; do
    python run.py --config-name timg_13 server=star04 algo=${algo}_step1500_tmax150k ++general.seed_offset=${seed};
    done; done;

# Fig. 12-14 reuse data from previous experiments

# Fig. 15: sensitivity analysis: lambda 12.5
## Fig. 15 (a)
configs=(28 27 26 25);
steps=(16660 5000 1660 500);
for i in "${!configs[@]}"; do
  for algo in pbt pb2rand bgpbt firepbt; do
    for seed in {0..4}; do
      python run.py --config-name c10_${configs[$i]} server=star04 algo=${algo}_step${steps[$i]}_lambda12_5 ++general.seed_offset=${seed};
    done; done; done

## Fig. 15 (b)
configs=(38 37 36 35);
steps=(50e6 15e6 5e6 15e5);
for i in "${!configs[@]}"; do
  for algo in pbt pb2rand pb2mix bgpbt firepbt; do
    for seed in {0..6}; do
      python run.py --config-name humanoid_${configs[$i]} server=star04 algo=${algo}_step${steps[$i]}_lambda12_5 ++general.seed_offset=${seed};
    done; done; done


# Fig. 16: change perturbation factor in FIRE-PBT
## Fig. 16 (a)
for seed in {0..6}; do
  python run.py --config-name humanoid_39 server=star04 ++general.seed_offset=${seed};
done;

## Fig. 16 (b)
for seed in {0..6}; do
  python run.py --config-name hopper_27 server=star04 ++general.seed_offset=${seed};
done;