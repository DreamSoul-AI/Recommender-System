python make.py --mode base --dataset ML100K --run tokenize --num_experiments 1 --round 1
python make.py --mode base --dataset ML100K --run train --num_experiments 1 --round 1
python make.py --mode base --dataset ML100K --run test --num_experiments 1 --round 1

python make.py --mode base --dataset ML1M --run tokenize --num_experiments 1 --round 1
python make.py --mode base --dataset ML1M --run train --num_experiments 1 --round 1
python make.py --mode base --dataset ML1M --run test --num_experiments 1 --round 1

python make.py --mode base --dataset Amazon --run tokenize --num_experiments 1 --round 1
python make.py --mode base --dataset Amazon --run train --num_experiments 1 --round 1
python make.py --mode base --dataset Amazon --run test --num_experiments 1 --round 1

python make.py --mode base --dataset Douban --run tokenize --num_experiments 1 --round 1
python make.py --mode base --dataset Douban --run train --num_experiments 1 --round 1
python make.py --mode base --dataset Douban --run test --num_experiments 1 --round 1