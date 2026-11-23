import time
import random
import os
import concurrent.futures
from numba import njit
import numpy as np

# =================================================================
# NÚCLEO MATEMÁTICO COMPILADO A LENGUAJE MÁQUINA (C++) CON NUMBA
# =================================================================

@njit
def calcular_aptitud_jit(tablero):
    n = len(tablero)
    conflictos = 0
    
    d1_counts = np.zeros(2 * n, dtype=np.int32)
    d2_counts = np.zeros(2 * n, dtype=np.int32)
    
    for c in range(n):
        f = tablero[c]
        d1_counts[f + c] += 1
        d2_counts[f - c + n] += 1
        
    for i in range(2 * n):
        if d1_counts[i] > 1:
            conflictos += (d1_counts[i] * (d1_counts[i] - 1)) // 2
        if d2_counts[i] > 1:
            conflictos += (d2_counts[i] * (d2_counts[i] - 1)) // 2
            
    return conflictos

@njit
def evaluar_intercambio_jit(tablero, c1, c2):
    n = len(tablero)
    f1 = tablero[c1]
    f2 = tablero[c2]
    
    ch_antes = 0
    for c in range(n):
        if c == c1 or c == c2: continue
        f = tablero[c]
        if f + c == f1 + c1 or f - c == f1 - c1: ch_antes += 1
        if f + c == f2 + c2 or f - c == f2 - c2: ch_antes += 1
        
    ch_despues = 0
    for c in range(n):
        if c == c1 or c == c2: continue
        f = tablero[c]
        if f + c == f2 + c1 or f - c == f2 - c1: ch_despues += 1
        if f + c == f1 + c2 or f - c == f1 - c2: ch_despues += 1
        
    if abs(f1 - f2) == abs(c1 - c2):
        ch_antes += 1
        
    if abs(f2 - f1) == abs(c1 - c2):
        ch_despues += 1

    return ch_antes - ch_despues


# =================================================================
# FUNCIÓN DE REPARACIÓN MULTIPROCESO (Ajustada para N=10000)
# =================================================================
def reparar_y_evaluar_acelerado(args):
    tablero_list, n, reinas_fijas, cols_libres, max_pasos = args
    tablero = np.array(tablero_list, dtype=np.int32)
    tabu_list = [] 
    
    for _ in range(max_pasos):
        d1_vistas = {}
        d2_vistas = {}
        conflictos_col = set()
        
        for c in range(n):
            f = tablero[c]
            d1_vistas.setdefault(f + c, []).append(c)
            d2_vistas.setdefault(f - c + n, []).append(c)
            
        for cols in d1_vistas.values():
            if len(cols) > 1: conflictos_col.update(cols)
        for cols in d2_vistas.values():
            if len(cols) > 1: conflictos_col.update(cols)
            
        cols_conf = [c for c in conflictos_col if c not in reinas_fijas]
        if not cols_conf: break
        
        c1 = random.choice(cols_conf)
        mejor_c2 = -1
        mejor_reduccion = 0
        
        # Ampliamos la muestra aleatoria a 30 candidatas para dar más margen de mejora en N=10000
        muestra_c2 = random.sample(cols_libres, min(30, len(cols_libres)))
        
        for c2 in muestra_c2:
            if c1 == c2: continue
            if (c1, c2) in tabu_list or (c2, c1) in tabu_list: continue
                
            reduccion = evaluar_intercambio_jit(tablero, c1, c2)
            
            if reduccion > mejor_reduccion:
                mejor_reduccion = reduccion
                mejor_c2 = c2
                
        if mejor_c2 != -1:
            tablero[c1], tablero[mejor_c2] = tablero[mejor_c2], tablero[c1]
            tabu_list.append((c1, mejor_c2))
            if len(tabu_list) > 10: tabu_list.pop(0)
        else:
            if random.random() < 0.1:
                cr = random.choice(cols_libres)
                tablero[c1], tablero[cr] = tablero[cr], tablero[c1]
                
    return tablero.tolist(), int(calcular_aptitud_jit(tablero))

# =================================================================
# CLASE PRINCIPAL (Bucle Indefinido)
# =================================================================
class MemeticoCompiladoIlimitado:
    def __init__(self, n, reinas_fijas, tam_poblacion=20):
        self.n = n
        self.reinas_fijas = reinas_fijas
        self.tam_poblacion = tam_poblacion
        self.filas_fijas = set(reinas_fijas.values())
        self.cols_libres = [c for c in range(n) if c not in reinas_fijas]
        self.filas_libres = [f for f in range(n) if f not in self.filas_fijas]

    def crear_ind(self):
        tablero = [-1] * self.n
        for c, f in self.reinas_fijas.items():
            tablero[c] = f
        filas = self.filas_libres[:]
        random.shuffle(filas)
        for c, f in zip(self.cols_libres, filas):
            tablero[c] = f
        return tablero

    def cruzar(self, p1, p2):
        hijo = [-1] * self.n
        for c, f in self.reinas_fijas.items(): hijo[c] = f
        punto = random.randint(0, len(self.cols_libres) - 1)
        heredadas = set(self.cols_libres[:punto])
        filas_usadas = set()
        
        for c in heredadas:
            hijo[c] = p1[c]
            filas_usadas.add(p1[c])
            
        idx = punto
        for c in self.cols_libres:
            if p2[c] not in filas_usadas:
                hijo[self.cols_libres[idx]] = p2[c]
                filas_usadas.add(p2[c])
                idx += 1
        return hijo

    def resolver(self):
        hilos = os.cpu_count() or 4
        print(f"[*] Compilador JIT Numba Activado. Multiproceso en {hilos} núcleos...")
        
        # Warm-up de compilación JIT
        _ = calcular_aptitud_jit(np.zeros(self.n, dtype=np.int32))
        _ = evaluar_intercambio_jit(np.zeros(self.n, dtype=np.int32), 0, 1)

        t_inicio_real = time.perf_counter()

        with concurrent.futures.ProcessPoolExecutor(max_workers=hilos) as executor:
            # En la población inicial asignamos 80 pasos de reparación profunda
            args_ini = [(self.crear_ind(), self.n, self.reinas_fijas, self.cols_libres, 80) for _ in range(self.tam_poblacion)]
            resultados = list(executor.map(reparar_y_evaluar_acelerado, args_ini))
            
            poblacion = [r[0] for r in resultados]
            aptitudes = [r[1] for r in resultados]
            
            mejor_hist = min(aptitudes)
            gens_sin_mejora = 0
            generacion = 0
            
            # BUCLE ILIMITADO: Se ejecutará hasta encontrar 0 conflictos
            while True:
                mejor_actual = min(aptitudes)
                if generacion % 5 == 0:
                    print(f"Gen {generacion:05d} -> Conflictos: {mejor_actual}")
                    
                if mejor_actual == 0:
                    return True, poblacion[aptitudes.index(0)], generacion, time.perf_counter() - t_inicio_real
                    
                if mejor_actual >= mejor_hist:
                    gens_sin_mejora += 1
                else:
                    mejor_hist = mejor_actual
                    gens_sin_mejora = 0
                    
                # REINICIO TÁCTICO si pasa 15 generaciones sin bajar el mínimo histórico
                if gens_sin_mejora >= 15:
                    print(f"🔄 REINICIO TÁCTICO (Gen {generacion}): Atascado {gens_sin_mejora} ciclos en {mejor_actual}. Regenerando población...")
                    indices = sorted(range(len(aptitudes)), key=lambda i: aptitudes[i])
                    nueva_pob = [poblacion[indices[0]], poblacion[indices[1]]]
                    
                    args_restart = [(self.crear_ind(), self.n, self.reinas_fijas, self.cols_libres, 80) for _ in range(self.tam_poblacion - 2)]
                    resultados_restart = list(executor.map(reparar_y_evaluar_acelerado, args_restart))
                    nueva_pob.extend([r[0] for r in resultados_restart])
                    
                    poblacion = nueva_pob
                    aptitudes = [r[1] for r in resultados_restart]
                    aptitudes.insert(0, mejor_actual)
                    aptitudes.insert(0, mejor_actual)
                    gens_sin_mejora = 0
                    generacion += 1
                    continue
                    
                indices = sorted(range(len(aptitudes)), key=lambda i: aptitudes[i])
                nueva_poblacion = [poblacion[indices[0]], poblacion[indices[1]]]
                
                args_cruce = []
                while len(nueva_poblacion) + len(args_cruce) < self.tam_poblacion:
                    t1 = random.sample(list(zip(poblacion, aptitudes)), 3)
                    p1 = min(t1, key=lambda x: x[1])[0]
                    t2 = random.sample(list(zip(poblacion, aptitudes)), 3)
                    p2 = min(t2, key=lambda x: x[1])[0]
                    
                    hijo = self.cruzar(p1, p2)
                    args_cruce.append((hijo, self.n, self.reinas_fijas, self.cols_libres, 35))
                    
                resultados_hijos = list(executor.map(reparar_y_evaluar_acelerado, args_cruce))
                
                for r in resultados_hijos: nueva_poblacion.append(r[0])
                    
                poblacion = nueva_poblacion
                aptitudes = [mejor_actual, mejor_actual] + [r[1] for r in resultados_hijos]
                generacion += 1


if __name__ == "__main__":
    N = 10000
    # 15 reinas fijas escaladas a posiciones proporcionales en el tablero de 10000
    filas_iniciales = [200, 500, 900, 1500, 2200, 2700, 3200, 3700, 4200, 4700, 5200, 5700, 6200, 6700, 7200]
    reinas_iniciales = {col * 100: fila for col, fila in enumerate(filas_iniciales)}

    print(f"--- ALGORITMO COMPILADO JIT ILIMITADO N={N} ---")
    print(f"Tablero de {N*N} casillas. Sin límite de generaciones.")
    
    motor = MemeticoCompiladoIlimitado(N, reinas_iniciales, tam_poblacion=20)
    
    exito, _, gens, t_ejecucion = motor.resolver()

    if exito:
        print(f"\n¡VICTORIA ABSOLUTA! Solución PERFECTA encontrada en {gens} generaciones.")
    print(f"Tiempo de ejecución puro: {t_ejecucion:.4f} segundos")