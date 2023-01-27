# Stochastic weak Minty variational inequalities

This code accompanies the paper titled [_Solving stochastic weak Minty variational inequalities without increasing batch size_](https://openreview.net/forum?id=ejR4E1jaH9k) at ICLR 2023.

## Installations

```bash
conda create -n swmvi python=3.8
conda activate swmvi
pip install -r requirements.txt
```

## Usage

To recreate the figures in the paper:

1. Run the experiment scripts in the section below.
2. Run `python plotting_script.py` to populate `/figs`.


## Experiments


```bash
# Bilinear
python runner.py --T 500000 --decrease sqrt --decrease-factor 100 --method P1BCSEG+ --noise 1.0 --gamma 0.5 --alpha0 0.055 --beta0 1 --problem bilinear --offset 0.9  --init 0.0 0.0 --projection linf --name "P1BCSEG+|bilinear|offset=0.9|projection=linf|alpha0=0.055|noise=1.0|gamma=0.5|decrease=sqrt-100|T=500000"
python runner.py --T 500000 --decrease sqrt --decrease-factor 100 --method P1SEG+ --noise 1.0 --gamma 0.5 --alpha0 0.055 --beta0 1 --problem bilinear --offset 0.9  --init 0.0 0.0 --projection linf --name "P1SEG+|bilinear|offset=0.9|projection=linf|alpha0=0.055|noise=1.0|gamma=0.5|decrease=sqrt-100|T=500000"
python runner.py --T 500000 --decrease sqrt --decrease-factor 100 --method P2SEG+ --noise 1.0 --gamma 0.5 --alpha0 0.055 --beta0 1 --problem bilinear --offset 0.9  --init 0.0 0.0 --projection linf --name "P2SEG+|bilinear|offset=0.9|projection=linf|alpha0=0.055|noise=1.0|gamma=0.5|decrease=sqrt-100|T=500000"
python runner.py --T 500000 --decrease sqrt --decrease-factor 100 --method P1SEG+ --noise 1.0 --gamma 0.1 --alpha0 0.055 --beta0 1 --problem bilinear --offset 0.9  --init 0.0 0.0 --projection linf --name "P1SEG+|bilinear|offset=0.9|projection=linf|alpha0=0.055|noise=1.0|gamma=0.1|decrease=sqrt-100|T=500000"

python runner.py --T 500000 --decrease linear --decrease-factor 100 --method P1BCSEG+ --noise 1.0 --gamma 0.5 --alpha0 0.055 --beta0 1 --problem bilinear --offset 0.9  --init 0.0 0.0 --projection linf --name "P1BCSEG+|bilinear|offset=0.9|projection=linf|alpha0=0.055|noise=1.0|gamma=0.5|decrease=linear-100|T=500000"
python runner.py --T 500000 --decrease linear --decrease-factor 100 --method P1SEG+ --noise 1.0 --gamma 0.5 --alpha0 0.055 --beta0 1 --problem bilinear --offset 0.9  --init 0.0 0.0 --projection linf --name "P1SEG+|bilinear|offset=0.9|projection=linf|alpha0=0.055|noise=1.0|gamma=0.5|decrease=linear-100|T=500000"
python runner.py --T 500000 --decrease linear --decrease-factor 100 --method P2SEG+ --noise 1.0 --gamma 0.5 --alpha0 0.055 --beta0 1 --problem bilinear --offset 0.9  --init 0.0 0.0 --projection linf --name "P2SEG+|bilinear|offset=0.9|projection=linf|alpha0=0.055|noise=1.0|gamma=0.5|decrease=linear-100|T=500000"
python runner.py --T 500000 --decrease linear --decrease-factor 1000 --method P1SEG+ --noise 1.0 --gamma 0.1 --alpha0 0.055 --beta0 1 --problem bilinear --offset 0.9  --init 0.0 0.0 --projection linf --name "P1SEG+|bilinear|offset=0.9|projection=linf|alpha0=0.055|noise=1.0|gamma=0.1|decrease=linear-1000|T=500000"

# Quadratic game
python runner.py --T 500000 --decrease sqrt --decrease-factor 100 --method EG+ --noise 0.1 --gamma 0.5 --alpha0 0.055 --beta0 1 --rho -0.1  --init 1.0 1.0 --name "EG+|rho=-0.1|alpha0=0.055|beta0=1|noise=0.1|gamma=0.5|decrease=sqrt-100|T=500000"
python runner.py --T 500000 --decrease sqrt --decrease-factor 100 --method BCSEG+ --noise 0.1 --gamma 0.5 --alpha0 0.055 --beta0 1 --rho -0.1  --init 1.0 1.0 --name "BCSEG+|rho=-0.1|alpha0=0.055|beta0=1|noise=0.1|gamma=0.5|decrease=sqrt-100|T=500000"
python runner.py --T 10000 --decrease sqrt --decrease-factor 100 --method SEG --noise 0.1 --gamma 0.5 --alpha0 0.055 --beta0 1 --rho -0.1  --init 1.0 1.0 --name "SEG|rho=-0.1|alpha0=0.055|beta0=1|noise=0.1|gamma=0.5|decrease=sqrt-100|T=10000"

python runner.py --T 500000 --decrease linear --decrease-factor 100 --method EG+ --noise 0.1 --gamma 0.5 --alpha0 0.055 --beta0 1 --rho -0.1  --init 1.0 1.0 --name "EG+|rho=-0.1|alpha0=0.055|noise=0.1|gamma=0.5|decrease=linear-100|T=500000"
python runner.py --T 500000 --decrease linear --decrease-factor 100 --method BCSEG+ --noise 0.1 --gamma 0.5 --alpha0 0.055 --beta0 1 --rho -0.1  --init 1.0 1.0 --name "BCSEG+|rho=-0.1|alpha0=0.055|noise=0.1|gamma=0.5|decrease=linear-100|T=500000"
python runner.py --T 500000 --decrease linear --decrease-factor 100 --method SEG --noise 0.1 --gamma 0.5 --alpha0 0.055 --beta0 1 --rho -0.1  --init 1.0 1.0 --name "SEG|rho=-0.1|alpha0=0.055|noise=0.1|gamma=0.5|decrease=linear-100|T=500000"

# GlobalForsaken
python runner.py --T 500000 --decrease sqrt --decrease-factor 100 --method P2EG+ --noise 0.1 --alpha0 0.055 --beta0 1 --init 1.0 1.0 --gamma 0.33 --problem GlobalForsaken --init 1.0 1.0 --name "GlobalForsaken|P2EG+|alpha0=0.055|beta0=1|noise=0.1|gamma=0.33|decrease=sqrt-100|T=500000"
python runner.py --T 500000 --decrease sqrt --decrease-factor 100 --method P1BCSEG+ --noise 0.1 --alpha0 0.055 --beta0 1 --init 1.0 1.0 --gamma 0.33 --problem GlobalForsaken --init 1.0 1.0 --name "GlobalForsaken|P1BCSEG+|alpha0=0.055|beta0=1|noise=0.1|gamma=0.33|decrease=sqrt-100|T=500000"
python runner.py --T 500000 --decrease sqrt --decrease-factor 100 --method P2SEG --noise 0.1 --alpha0 0.055 --beta0 1 --init 1.0 1.0 --gamma 0.33 --problem GlobalForsaken --init 1.0 1.0 --name "GlobalForsaken|P2SEG|alpha0=0.055|beta0=1|noise=0.1|gamma=0.33|decrease=sqrt-100|T=500000"

python runner.py --T 500000 --decrease linear --decrease-factor 100 --method P2EG+ --noise 0.1 --alpha0 0.055 --beta0 1 --init 1.0 1.0 --gamma 0.33 --problem GlobalForsaken --init 1.0 1.0 --name "GlobalForsaken|P2EG+|alpha0=0.055|noise=0.1|gamma=0.33|decrease=linear-100|T=500000"
python runner.py --T 500000 --decrease linear --decrease-factor 100 --method P1BCSEG+ --noise 0.1 --alpha0 0.055 --beta0 1 --init 1.0 1.0 --gamma 0.33 --problem GlobalForsaken --init 1.0 1.0 --name "GlobalForsaken|P1BCSEG+|alpha0=0.055|noise=0.1|gamma=0.33|decrease=linear-100|T=500000"
python runner.py --T 500000 --decrease linear --decrease-factor 100 --method P2SEG --noise 0.1 --alpha0 0.055 --beta0 1 --init 1.0 1.0 --gamma 0.33 --problem GlobalForsaken --init 1.0 1.0 --name "GlobalForsaken|P2SEG|alpha0=0.055|noise=0.1|gamma=0.33|decrease=linear-100|T=500000"

# Constrained quadratic
python runner.py --T 500000 --decrease sqrt --decrease-factor 100 --method P1SEG+ --noise 0.1 --gamma 0.5 --alpha0 0.055 --beta0 1 --rho -0.1 --problem quadratic --offset 0.9  --init 0.8 0.8 --projection linf --name "P1SEG+|rho=-0.1|offset=0.9|projection=linf|alpha0=0.055|beta0=1|noise=0.1|gamma=0.5|decrease=sqrt-100|T=500000"
python runner.py --T 500000 --decrease sqrt --decrease-factor 100 --method P1SEG+ --noise 0.1 --gamma 0.1 --alpha0 0.055 --beta0 1 --rho -0.1 --problem quadratic --offset 0.9  --init 0.8 0.8 --projection linf --name "P1SEG+|rho=-0.1|offset=0.9|projection=linf|alpha0=0.055|beta0=1|noise=0.1|gamma=0.1|decrease=sqrt-100|T=500000"
python runner.py --T 500000 --decrease sqrt --decrease-factor 100 --method P1SEG+ --noise 0.1 --gamma 0.01 --alpha0 0.055 --beta0 1 --rho -0.1 --problem quadratic --offset 0.9  --init 0.8 0.8 --projection linf --name "P1SEG+|rho=-0.1|offset=0.9|projection=linf|alpha0=0.055|beta0=1|noise=0.1|gamma=0.01|decrease=sqrt-100|T=500000"
python runner.py --T 500000 --decrease sqrt --decrease-factor 100 --method P1BCSEG+ --noise 0.1 --gamma 0.5 --alpha0 0.055 --beta0 1 --rho -0.1 --problem quadratic --offset 0.9  --init 0.8 0.8 --projection linf --name "P1BCSEG+|rho=-0.1|offset=0.9|projection=linf|alpha0=0.055|beta0=1|noise=0.1|gamma=0.5|decrease=sqrt-100|T=500000"

python runner.py --T 500000 --decrease linear --decrease-factor 1000 --method P1SEG+ --noise 0.1 --gamma 0.5 --alpha0 0.055 --beta0 1 --rho -0.1 --problem quadratic --offset 0.9  --init 0.8 0.8 --projection linf --name "P1SEG+|rho=-0.1|offset=0.9|projection=linf|alpha0=0.055|beta0=1|noise=0.1|gamma=0.5|decrease=linear-1000|T=500000"
python runner.py --T 500000 --decrease linear --decrease-factor 1000 --method P1SEG+ --noise 0.1 --gamma 0.1 --alpha0 0.055 --beta0 1 --rho -0.1 --problem quadratic --offset 0.9  --init 0.8 0.8 --projection linf --name "P1SEG+|rho=-0.1|offset=0.9|projection=linf|alpha0=0.055|beta0=1|noise=0.1|gamma=0.1|decrease=linear-1000|T=500000"
python runner.py --T 500000 --decrease linear --decrease-factor 5000 --method P1SEG+ --noise 0.1 --gamma 0.01 --alpha0 0.055 --beta0 1 --rho -0.1 --problem quadratic --offset 0.9  --init 0.8 0.8 --projection linf --name "P1SEG+|rho=-0.1|offset=0.9|projection=linf|alpha0=0.055|beta0=1|noise=0.1|gamma=0.01|decrease=linear-5000|T=500000"
python runner.py --T 500000 --decrease linear --decrease-factor 1000 --method P1BCSEG+ --noise 0.1 --gamma 0.5 --alpha0 0.055 --beta0 1 --rho -0.1 --problem quadratic --offset 0.9  --init 0.8 0.8 --projection linf --name "P1BCSEG+|rho=-0.1|offset=0.9|projection=linf|alpha0=0.055|beta0=1|noise=0.1|gamma=0.5|decrease=linear-1000|T=500000"
```
