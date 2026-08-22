# Notación — `arfield`

Diccionario de símbolos. Existe porque hay **cuatro** notaciones en juego y no
coinciden entre sí:

1. El Placko & Kundu, que además es multi-autor y deriva la notación entre capítulos
2. El código de Baresch (`actweez_col`), en MATLAB y con comentarios en francés
3. El Phys. Rev. Applied 18, 034026
4. El código de `arfield`

Regla: **cada vez que se traduzca un símbolo entre dos de estas fuentes, se anota
aquí.** Verificar caso por caso contra el código o el texto, nunca contra la memoria.

Estado de cada fila:
- ✅ verificado contra la fuente
- ⚠️ pendiente de verificar
- ❌ discrepancia conocida, sin resolver

---

## 1. Convenios base

| | Valor | Estado |
|---|---|---|
| Factor temporal | `e^{-iωt}` | ✅ |
| Función de Green | `G(R) = e^{ikR} / (4πR)` | ✅ |
| Velocidad desde presión | `v = ∇p / (iωρ)` | ✅ |
| Potencial de velocidad | `φ = p / (iωρ)`, con `v = ∇φ` | ✅ |
| Unidades | SI en todo el código | — |
| dtype de campos acústicos | `complex128` | — |

**Por qué estos tres van juntos.** El factor temporal es una elección libre de
contabilidad; una vez fijado, la física determina los otros dos.

- Que la onda **salga** exige que los signos de `kR` y `ωt` sean opuestos en la
  fase total. Con `e^{-iωt}`, la fase de una fuente puntual es `kR − ωt`; siguiendo
  una cresta, `dR/dt = ω/k > 0`. Sale. Por eso la Green lleva `e^{+ikR}`.
- Euler linealizado `ρ ∂v/∂t = −∇p` con `∂/∂t → −iω` da `−iωρ v = −∇p`, de donde
  `v = ∇p/(iωρ)`.

Bajo el convenio contrario (`e^{+iωt}`) toda la solución es el **conjugado complejo**
de esta. `|p|` y `|∇p|` son idénticos, y por tanto `U` de Gor'kov y la fuerza también.
Por eso un error de convenio es invisible hasta el momento de conectar con código
ajeno (Hito 6).

**Verificación numérica**: `np.angle(G(R₂)) − np.angle(G(R₁))` debe dar `+k(R₂−R₁)`.

Registro completo en `docs/adr/0001-time-convention.md`.

### Concordancia entre fuentes

| Fuente | Evidencia | Convenio | Estado |
|---|---|---|---|
| Placko & Kundu | Ecs. 1.19–1.20; Green de la Ec. 1.17 con `e^{+ikr}` | `e^{-iωt}` | ✅ |
| Baresch, `plane_wave.m` | factor `4*pi*1i^(n)` — firma de la expansión de `e^{+ik·r}` | `e^{-iωt}` | ✅ |
| Baresch, `focused_beam.m` | usa `sbesselh1_diego` (Hankel de 1ª especie) para la radiación saliente | `e^{-iωt}` | ✅ |
| PRApplied 18, 034026 | relación `φ = p/(iωρ)` | `e^{-iωt}` | ✅ |

Tres fuentes independientes concuerdan.

### Qué fija y qué no fija el convenio temporal

Fija:

- el signo del exponente de la Green — la fase debe **avanzar** con la distancia
- el signo entre velocidad y gradiente de presión, vía Euler y `∂/∂t → −iω`
- qué Hankel esférica es la saliente
- el **sentido azimutal** de la fase `e^{imφ}` en la expansión en armónicos

**No** fija:

- la **normalización** de los armónicos esféricos (factores reales positivos)
- la fase de **Condon–Shortley**, `(−1)^m` (un signo global)

Son convenios independientes. Se resuelven en el Hito 6 contra el código de
Baresch, no por deducción.

### Hankel: cuál se usa dónde

Las Hankel esféricas son las combinaciones complejas de las dos soluciones reales
de la parte radial: `h_n^(1) = j_n + i·y_n`, `h_n^(2) = j_n − i·y_n`. Son
conjugadas entre sí. Asintóticamente,

```
h_n^(1)(kr) ≈ (−i)^(n+1) · e^{+ikr} / (kr)     saliente bajo e^{-iωt}
h_n^(2)(kr) ≈ (+i)^(n+1) · e^{−ikr} / (kr)     entrante
```

Los prefactores son constantes y no mueven crestas; decide el exponente.

**Ancla exacta** para `n = 0`, que hace la elección evidente:

```
h_0^(1)(x) = −i·e^{ix}/x        ⟹     G(R) = (ik/4π)·h_0^(1)(kR)
```

Despeje: `e^{ix}/x = h_0^(1)(x)/(−i) = i·h_0^(1)(x)`, usando `1/(−i) = i`. La `k`
aparece y se cancela porque la Hankel come el argumento adimensional `kR` mientras
la Green tiene `R` con dimensiones en el denominador. **La función de Green *es* la
Hankel de primera especie de orden cero**, salvo constante.

| Campo | Radial | Por qué |
|---|---|---|
| Incidente, alrededor del centro de la partícula | `j_n` | Finito en `r = 0`. `j_n = (h^(1) + h^(2))/2`: saliente + entrante = estacionaria |
| Dispersado por la partícula | `h_n^(1)` | Nace en la partícula y se va al infinito |
| — | `h_n^(2)` | **No aparece nunca en este proyecto.** No hay fuentes en el infinito radiando hacia adentro |

---

## 2. Símbolos del DPSM

| Símbolo | Significado | En `arfield` |
|---|---|---|
| `k` | número de onda, `ω/c` | `k` |
| `ω` | frecuencia angular, `2πf` | `omega` |
| `ρ` | densidad del medio | `rho` |
| `c` | velocidad del sonido en el medio | `c` |
| `λ` | longitud de onda, `2π/k` | `wavelength` |
| `p` | presión acústica (compleja) | `p` |
| `v` | velocidad de partícula (vector complejo) | `v` |
| `v₀` | amplitud de la velocidad normal prescrita en la cara del transductor | `v0` |
| `A` | vector de intensidades de las fuentes puntuales — **las incógnitas** | `A` |
| `V` | vector de condiciones de frontera | `V` |
| `r_s` | distancia a la que las fuentes se colocan **detrás** de la superficie | `r_s` |
| `ΔS_m` | área del elemento de malla | `areas` |
| `n̂` | normal unitaria saliente en el punto de colocación | `normals` |
| `M` | número de puntos objetivo (observación o colocación) | `n_targets` |
| `N` | número de fuentes puntuales | `n_sources` |
| `a` | radio del pistón circular | `radius` |

### Las matrices del libro

Verificado contra las Ecs. 1.25j y 2.19:

```
P_T = Q_TS · A_S        Q = e^{ikr}/r                          presión      ✅
V_T = M_TS · A_S        M = (1/(iωρ))·(x₃/r³)(ikr−1)e^{ikr}    velocidad    ✅
```

Que `M` es `Q` derivada y proyectada se comprueba a mano:

```
∂/∂x₃ [ e^{ikr}/r ] = x₃ (ikr − 1) e^{ikr} / r³        con ∂r/∂x₃ = x₃/r
```

y dividiendo por `iωρ` sale la `M` del libro término a término. El `x₃/r³` es la
proyección para una superficie de normal `ê₃`; en el caso general el factor es
`(R·n̂)/r³`.

**Subíndice doble = (target, source)**, en ese orden. Primero dónde se evalúa,
segundo de dónde vienen las fuentes. Coincide con la forma `(M, N)` de `arfield`.

### ⚠️ Tres colisiones de notación del libro

1. **`M` es la matriz de velocidad y el número de puntos objetivo**, en el mismo
   párrafo. Regla de desambiguación: `M` con dos letras de subíndice (`M_SS`,
   `M_TS`) es matriz; `M` desnuda es un conteo.
2. **Los índices minúsculos van cruzados**: el libro escribe las fuentes como `y_m`
   con `m = 1…N`, y los objetivos como `x_n` con `n = 1…M`. En `arfield` los
   índices de `einsum` usan `m` para observación y `n` para fuente — **al revés
   que el libro**. Anotado a propósito.
3. **`V` es el vector de condiciones de frontera en DPSM y el volumen de la
   partícula en Gor'kov.** En el código nunca se llaman igual: `V` y `volume`.

### ⚠️ El factor 4π

La `Q` del libro es `e^{ikr}/r`, **sin** el `1/(4π)`; el libro lo absorbe en las
`A`. El `green_TS` de `arfield` **sí** lo lleva, por el convenio de la §1.

Misma cantidad física, escala distinta. Consecuencia práctica: `p` sale igual,
pero **los valores de `A` no son comparables directamente con los del libro**.

### ⚠️ «Matriz de influencia» no es del libro

Placko & Kundu nunca bautizan estas matrices. Las describen: «la matriz que
relaciona los vectores `V_S` y `A_S`». El índice analítico del final no tiene
entradas para `M`, `Q` ni «influence matrix».

El término es del proyecto, no una traducción.

### Dos usos distintos de `A`

| Formulación | Cómo se obtienen las `A` |
|---|---|
| Rayleigh–Sommerfeld discretizado | **Asignadas**: `A_m = B·ΔS_m/4π` con `B = −2iωρv₀`. Sin sistema lineal. |
| DPSM | **Incógnitas**: salen de resolver el sistema imponiendo la condición de frontera |

Las dos aparecen en la Fig. 1.35 del libro, y en el Hito 1 se calculan ambas. No
confundirlas: la primera solo vale para pistón plano en pantalla infinita.

### Fuentes triplete — no se usan ✅

El libro las introduce cuando hay que igualar las **tres** componentes de
velocidad: `3N` ecuaciones, y cada fuente se **sustituye** por tres fuentes
puntuales físicas con tres intensidades distintas, en los vértices de un triángulo
isósceles orientado al azar. `M_SS` queda `(3N × 3N)`.

Ese es el caso **viscoso**. El libro dice explícitamente que en fluido perfecto no
viscoso basta la componente normal y las dimensiones bajan de `3N` a `N`, con
fuentes puntuales simples.

Aire a 40 kHz es no viscoso a estos efectos. **`arfield` nunca usa tripletes.**

---

## 3. Nombres y formas en `arfield`

Cerrados en el ADR 0002. `M` = puntos objetivo, `N` = fuentes. Siempre en ese orden.

| Nombre | Forma | Contenido |
|---|---|---|
| `green_TS` | `(M, N)` | `G(R) = e^{ikR}/(4πR)` tabulada |
| `grad_green_TSj` | `(M, N, 3)` | `∇G`, componentes cartesianas en el tercer eje |
| `euler_gradn_green_TS` | `(M, N)` | `n̂·∇G/(iωρ)` — la matriz del sistema lineal |
| posiciones | `(N, 3)` o `(M, 3)` | puntos o fuentes, lista plana |
| `A` | `(N,)` | intensidades |
| `V` | `(M,)` | condiciones de frontera |

### Reglas de construcción del nombre

- **El nombre dice lo que el array *contiene***, nunca lo que produce al
  multiplicarlo. Por eso `vn_TS` (velocidad normal) está descartado.
- **`T` = target, `S` = source**, en ese orden, reflejando `(M, N)`. Permanente.
- **`j` = índice de componente cartesiana.** Su presencia marca el tercer eje.
  Misma letra que el libro usa en `v_j^n`.
- **Palabras en `snake_case`; subíndices pegados al final, en mayúscula.** La
  mayúscula separa palabras de índices, y distingue índices de superficie (`T`,
  `S`) del índice de componente (`j`).
- `euler` = pasó por `ρ ∂v/∂t = −∇p`, que es lo único que introduce `1/(iωρ)`.
- `gradn` = gradiente proyectado sobre la normal.

Nombres empiezan en minúscula a propósito: la regla `N806` de Ruff (`pep8-naming`)
marcaría `G_TS`. Verificado — `N` no está en el conjunto por defecto de Ruff, pero
esto deja la puerta abierta a activarla.

### Equivalencias

| `arfield` | Libro | Nota |
|---|---|---|
| `green_TS` | `Q_TS` | difiere en el factor `1/(4π)` |
| `grad_green_TSj` | — | el libro no lo nombra: va directo a la proyección |
| `euler_gradn_green_TS` | `M_TS` | equivalentes salvo el mismo `1/(4π)` |

---

## 4. Símbolos de Gor'kov

| Símbolo | Significado | En `arfield` |
|---|---|---|
| `U` | potencial de Gor'kov | `U` |
| `F` | fuerza de radiación, `F = −∇U` | `F` |
| `V` | **volumen de la partícula** — colisión con el `V` del DPSM | `volume` |
| `f₁` | factor de monopolo, `1 − κ_p/κ₀` | `f1` |
| `f₂` | factor de dipolo, `2(ρ_p−ρ₀)/(2ρ_p+ρ₀)` | `f2` |
| `κ` | compresibilidad; `κ₀` medio, `κ_p` partícula | `kappa_0`, `kappa_p` |
| `ρ_p` | densidad de la partícula | `rho_p` |

Aquí es donde `grad_green_TSj` completo hace falta: `|v|²` necesita las tres
componentes, no la proyección normal.

---

## 5. Símbolos de armónicos esféricos / GLMT

Pendiente de completar en el Hito 6. Verificado del código de Baresch:

| Elemento | En `actweez_col` | Nota | Estado |
|---|---|---|---|
| Coeficientes de expansión | `Anm` | vector disperso indexado por `ci` | ✅ |
| Índice combinado | `ci = n*(n+1) + m` | `combined_index.m`, del *Optical Tweezers Toolbox* (UQ, 2006) | ✅ |
| Truncamiento | `Nmax` | | ✅ |
| Hankel esférica 1ª especie | `sbesselh1_diego.m` | la saliente bajo `e^{-iωt}` | ✅ |
| Elevación / azimut | `alpha` / `beta` | comentarios en francés | ✅ |

### ❌ Discrepancia abierta — normalización de armónicos

Las instrucciones afirmaban que el paper usa armónicos **sin normalizar**
(`Y_nm = P_n^m(cosθ)·e^{imφ}`). El código dice otra cosa: en `plane_wave.m`,

```matlab
Pnm = legendre(n,cos(alpha),'norm');
Pnm = sqrt(2/(4*pi))*Pnm;   % rectificación de la normalización 'norm'
```

Baresch parte de la normalización de MATLAB y la **rectifica a armónicos
completamente normalizados**. Sus `Anm` están definidos contra esa base.

También hay tratamiento explícito de `m < 0`:

```matlab
if m<0
    Anm(ii) = Anm(ii)*(-1)^m;
end
```

que hay que rastrear hasta Condon–Shortley antes de comparar nada.
**No resolver antes del Hito 6.**

### ⚠️ Trampa de magnitud

El paper expande el **potencial de velocidad `φ`**, no la presión. El DPSM entrega
`p`. El factor de conversión es constante (`φ = p/(iωρ)`) pero olvidarlo escala la
fuerza. Verificar antes de comparar contra la Fig. 1(a).

---

## 6. Estructura del código de Baresch

```
actweez_col/
├── functions/              # el código propio del paper
│   ├── plane_wave.m        # A_nm de una onda plana
│   ├── focused_beam.m      # A_nm de un pistón cóncavo con bafle
│   ├── T_matrix.m          # esfera elástica
│   ├── T_matrix_Fluid.m    # esfera fluida
│   ├── force_X/Y/Z.m       # fuerza por componente
│   ├── translate_z_scalar.m
│   └── sbessel*_diego.m
├── examples/
│   ├── force_mie.m
│   ├── force_tweezers.m
│   ├── incident_fields.m
│   └── scattering_bubble.m
└── optical_tweezers_tbox/  # Optical Tweezers Toolbox (UQ 2006), dependencia
```

`optical_tweezers_tbox` es una librería de terceros, óptica, reutilizada por su
maquinaria de armónicos vectoriales e índices. No es código de Baresch.

---

## 7. Cómo usar el libro

Volumen editado multi-autor. La misma maquinaria se re-deriva con notación distinta.
Si una derivación no cierra, buscar lo mismo contado por otro autor suele destrabarla.

| Capítulos | Autores |
|---|---|
| 1–2 | Placko & Kundu |
| 3 | Kundu, Ahmad, Alnuaimi & Placko |
| 4 | Banerjee & Kundu |
| 5–6 | Liebeaux & Placko |
| 7–8 | Lissorgues, Cruau & Placko |
| 9–10 | Lemistre & Placko |
| 11 | Cruau & Placko |

**§11.2 tiene el glosario de los propios autores** — *medium, object, interface,
boundary conditions, frontier, workspace*.

Usar el índice analítico del final (p. 369), no el de contenidos. **Aviso**: ese
índice no tiene entradas para `M`, `Q` ni «influence matrix»; la única entrada
`Matrix, 274, 301, 360` apunta a los capítulos 7 y 11, no a la formulación.

---

## 8. Registro de traducciones

| Fecha | Símbolo | De | A | Nota |
|---|---|---|---|---|
| 22 ago 2026 | factor temporal | instrucciones (`e^{+iωt}`) | `e^{-iωt}` | La versión anterior del documento estaba mal; incompatible con `φ = p/(iωρ)` |
| 22 ago 2026 | `Q_TS` (libro) | `e^{ikr}/r` | `green_TS` = `e^{ikR}/(4πR)` | Difieren en `1/(4π)`; las `A` no son comparables |
| 22 ago 2026 | `M_TS` (libro) | `(1/(iωρ))(x₃/r³)(ikr−1)e^{ikr}` | `euler_gradn_green_TS` | Mismo objeto salvo el `1/(4π)` |
| 22 ago 2026 | índices `m`, `n` | libro: `m`=fuente, `n`=objetivo | `arfield`: `m`=objetivo, `n`=fuente | Cruzados a propósito, para que coincidan con `(M, N)` |
| 22 ago 2026 | «matriz de influencia» | — | terminología del proyecto | El libro nunca las bautiza |
| 22 ago 2026 | normalización de armónicos | atribuida al convenio temporal | convenio independiente | El temporal solo fija el sentido azimutal de `e^{imφ}` |
