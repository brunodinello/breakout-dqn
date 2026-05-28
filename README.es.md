# Breakout DQN

### Una red neuronal que aprende a jugar Atari Breakout mirando sólo los píxeles — sin reglas, sin pistas, con una recompensa como única guía.

> 🇬🇧 [English version](README.md)

![python](https://img.shields.io/badge/python-3.12-blue) ![pytorch](https://img.shields.io/badge/pytorch-2.x-ee4c2c) ![license](https://img.shields.io/badge/license-MIT-green)

---

![Progresión Double DQN](assets/double_dqn_progression.gif)

**Es el mismo agente en los episodios 0, 2000, 5000 y 9000.** Nadie le dijo qué es una pelota, para qué sirve la paleta ni que romper ladrillos da puntos. Lo único que vio fueron los píxeles de la pantalla y un número subiendo cuando acertaba. Para el episodio 9000 ya había descubierto, solo, la estrategia del túnel — el mismo truco que hizo famoso al paper original de DeepMind.

## Qué está haciendo

Un **agente de aprendizaje por refuerzo** es un programa que decide qué hacer, ve qué pasó, y se ajusta — el mismo bucle que usás vos la primera vez que agarrás un joystick para un juego que nunca jugaste.

En cada cuadro este agente mira la pantalla y elige uno de cuatro movimientos: quedarse quieto, sacar la pelota, izquierda, derecha. Por cada ladrillo que rompe recibe una pequeña recompensa, y esa es la *única* señal que recibe. Las reglas del Breakout, la idea de que hay una pelota, que conviene devolverla con la paleta — todo eso lo tiene que descubrir jugando.

"Entrenar", acá, quiere decir dejarlo jugar millones de cuadros mientras una red neuronal convolucional aprende a estimar, mirando directamente los píxeles, qué tan buena es cada acción en cada situación. Los primeros cientos de episodios parecen literalmente un generador aleatorio agarrando el joystick. Unos miles más adelante, está apuntando a la esquina a propósito.

## Qué hay acá adentro

- **DQN clásico** — la versión base del paper de Nature 2015 ([Mnih et al.](https://www.nature.com/articles/nature14236))
- **Double DQN** — la corrección de 2016 al sesgo de sobreestimación de los valores Q ([van Hasselt et al.](https://arxiv.org/abs/1509.06461))
- **Pipeline de preprocesamiento Atari completo** — grayscale, redimensión a 84×84, apilado de 4 cuadros, recorte de recompensas, ε-greedy con annealing
- **Checkpoints ya entrenados** — podés ver al agente jugar en 30 segundos, sin entrenar nada
- **GIFs de progresión** comparando ambos agentes
- **Reproducible** — entorno conda, requirements de pip, entrenamiento y evaluación con un solo comando, logging opcional a Weights & Biases

## Y el otro también

**DQN clásico**

![Progresión DQN](assets/dqn_progression.gif)

## Probalo en 30 segundos

```bash
conda env create -f environment.yml
conda activate breakout-dqn

python scripts/evaluate.py \
    --checkpoint checkpoints/double_dqn_final.dat \
    --episodes 5 --render
```

Si no usás conda, `pip install -r requirements.txt` también sirve.

## Entrená tu propio agente

```bash
python scripts/train.py \
    --agent double_dqn \
    --episodes 10000 \
    --video-dir videos/run1 --record-every 500 \
    --wandb
```

Las perillas que probablemente vas a querer tocar: `--agent {dqn,double_dqn}`, `--lr`, `--gamma`, `--epsilon-anneal-steps`, `--sync-target`. La lista completa está en `python scripts/train.py --help`.

Para regenerar los GIFs de progresión:

```bash
python scripts/make_gifs.py \
    --run videos/ddqn_training \
    --out assets/double_dqn_progression.gif \
    --episodes 0 2000 5000 9000
```

## Cómo funciona por dentro

1. **Preprocesamiento** — cada cuadro pasa a escala de grises, se redimensiona a 84×84, y se apilan los últimos 4 para que la red pueda percibir movimiento. Las recompensas se recortan a `{-1, 0, +1}`.
2. **Red** — una CNN pequeña (dos capas convolucionales más una totalmente conectada) devuelve un valor Q por cada acción.
3. **Experience replay** — las transiciones `(s, a, r, done, s')` se guardan en un buffer circular y se muestrean minibatches al azar, así el entrenamiento no queda sesgado por jugadas consecutivas.
4. **Exploración ε-greedy** — ε baja linealmente de 1.0 a 0.1 durante el primer millón de pasos, así el agente explora a fondo al principio y va confiando cada vez más en lo aprendido.
5. **Red objetivo (sólo Double DQN)** — una copia congelada de la red online evalúa el valor Q del próximo estado, mientras que la online elige *qué* acción evaluar. Separar esos dos roles es lo que elimina el sesgo de sobreestimación sistemático del DQN clásico.

## Estructura del proyecto

```
src/         # agentes, CNN, replay buffer, wrappers del entorno
scripts/     # train.py, evaluate.py, make_gifs.py
notebooks/   # train_and_evaluate.ipynb (flujo completo)
checkpoints/ # pesos .dat de los modelos entrenados
assets/      # GIFs del README
```

## Referencias

- Mnih et al., *Human-level control through deep reinforcement learning*, Nature 2015.
- van Hasselt, Guez, Silver, *Deep Reinforcement Learning with Double Q-learning*, AAAI 2016.
- [Gymnasium](https://gymnasium.farama.org/) para el entorno Atari, [PyTorch](https://pytorch.org/) para la red.

## Autor

**Bruno Dinello** — [GitHub](https://github.com/brunodinello) · [LinkedIn](https://www.linkedin.com/in/bruno-dinello)

## Licencia

MIT — ver [LICENSE](LICENSE).
