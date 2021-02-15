SHELL=/bin/bash
PROJECT_NAME=diygym4
python=/home/wassname/anaconda/envs/diygym4/bin/python
LOGURU_LEVEL=INFO

run:
	ulimit -S -m 65000000
	ulimit -S -v 65000000
	LOGURU_LEVEL=INFO \
	${python} main.py \
	--load auto \
	--demonstrations /media/wassname/Storage5/projects2/3ST/diy_bullet_conveyor/agents/pytorch-soft-actor-critic/data/rllib_demo

play:
	${python} \
		-m pdb -c continue \
		main.py \
			--load auto \
			--render \
			--debug \
			--eval \
			--stop-iters 600

pdb:
	# debug single thread
	ulimit -S -m 65000000
	ulimit -S -v 65000000
	${python} -m pdb main.py \
	-d \
	# --load /home/wassname/ray_results/SAC/SAC_ApplePick-v0_832b7_00000_0_2021-02-11_08-17-23/checkpoint_20/checkpoint-20
	# --demonstrations /media/wassname/Storage5/projects2/3ST/diy_bullet_conveyor/agents/pytorch-soft-actor-critic/data/rllib_demo


debug:
	# ray multithread debug, also need to run `ray debug` in another term
	ulimit -S -m 65000000
	ulimit -S -v 65000000
	RAY_PDB=1 \
		${python} main.py \
		# -d /media/wassname/Storage5/projects2/3ST/diy_bullet_conveyor/agents/pytorch-soft-actor-critic/data/rllib_demo

tb:
	${python} -m tensorboard --logdir=~/ray_results

## Export project requirements in multiple formats
doc_reqs:
	conda env export --no-builds --from-history --name $(PROJECT_NAME) > requirements/environment.min.yaml
	conda env export  --name $(PROJECT_NAME) > requirements/environment.max.yaml
	$(python) -m pip freeze > requirements/requirements.txt
