import time
import random
import os
import concurrent.futures
from numba import njit
import numpy as np

# =================================================================
# 1. PREPROCESADOR AC-3 Y CONSTRUCTOR MRV CON MIN-CONFLICTS FALLBACK
# =================================================================

def preprocesar_dominios_ac3(n, reinas_fijas):
    """
    Poda AC-3 / Forward Checking de paso único respecto a las reinas fijas.
    Calcula los dominios válidos para cada columna libre.
    """
    filas_bloqueadas = set(reinas_fijas.values())
    d1_bloqueadas = {f + c for c, f in reinas_fijas.items()}
    d2_bloqueadas = {f - c for c, f in reinas_fijas.items()}
    
    dominios = {}
    cols_libres = [c for c in range(n) if c not in reinas_fijas]
    
    for c in cols_libres:
        dom_c = [
            f for f in range(n) 
            if f not in filas_bloqueadas 
            and (f + c) not in d1_bloqueadas 
            and (f - c) not in d2_bloqueadas
        ]
        dominios[c] = dom_c
        
    return dominios, cols_libres, d1_bloqueadas, d2_bloqueadas


def generar_individuo_ac3_mrv(n, reinas_fijas, dominios, cols_libres, d1_bloqueadas, d2_bloqueadas):
    """
    Genera un individuo asignando primero las columnas más restringidas (MRV).
    Si una columna agota sus opciones del filtro AC-3, se aplica el Min-Conflicts Fallback.
    """
    tablero = [-1] * n
    for c, f in reinas_fijas.items():
        tablero[c] = f
        
    filas_disponibles = set(range(n)) - set(reinas_fijas.values())
    
    # Ordenación MRV con ruido aleatorio para desempate
    cols_ordenadas_mrv = sorted(
        cols_libres, 
        key=lambda c: (len(dominios[c]), random.random())
    )
    
    for c in cols_ordenadas_mrv:
        opciones_validas = [f for f in dominios[c] if f in filas_disponibles]
        
        if opciones_validas:
            f_elegida = random.choice(opciones_validas)
        else:
            # Fallback de mínimo conflicto en O(1) respecto a las diagonales fijas
            muestra_filas = random.sample(list(filas_disponibles), min(10, len(filas_disponibles)))
            f_elegida = min(muestra_filas, key=lambda f: (
                (1 if (f + c) in d1_bloqueadas else 0) + 
                (1 if (f - c) in d2_bloqueadas else 0)
            ))
            
        tablero[c] = f_elegida
        filas_disponibles.remove(f_elegida)
        
    return tablero


# =================================================================
# 2. NÚCLEO MATEMÁTICO COMPILADO A C++ (NUMBA JIT)
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
def evaluar_intercambio_gls_jit(tablero, c1, c2, pen_d1, pen_d2, lambda_pen):
    n = len(tablero)
    f1, f2 = tablero[c1], tablero[c2]
    
    ch_antes = 0
    p_antes = pen_d1[f1 + c1] + pen_d2[f1 - c1 + n] + pen_d1[f2 + c2] + pen_d2[f2 - c2 + n]
    
    for c in range(n):
        if c == c1 or c == c2: continue
        f = tablero[c]
        if f + c == f1 + c1 or f - c == f1 - c1: ch_antes += 1
        if f + c == f2 + c2 or f - c == f2 - c2: ch_antes += 1
        
    ch_despues = 0
    p_despues = pen_d1[f2 + c1] + pen_d2[f2 - c1 + n] + pen_d1[f1 + c2] + pen_d2[f1 - c2 + n]
    
    for c in range(n):
        if c == c1 or c == c2: continue
        f = tablero[c]
        if f + c == f2 + c1 or f - c == f2 - c1: ch_despues += 1
        if f + c == f1 + c2 or f - c == f1 - c2: ch_despues += 1
        
    if abs(f1 - f2) == abs(c1 - c2): ch_antes += 1
    if abs(f2 - f1) == abs(c1 - c2): ch_despues += 1

    costo_antes = ch_antes + lambda_pen * p_antes
    costo_despues = ch_despues + lambda_pen * p_despues
    return costo_antes - costo_despues


@njit
def evaluar_3swap_jit(tablero, c1, c2, c3):
    n = len(tablero)
    f1, f2, f3 = tablero[c1], tablero[c2], tablero[c3]
    
    ch_antes = 0
    for c in range(n):
        if c == c1 or c == c2 or c == c3: continue
        f = tablero[c]
        if f + c == f1 + c1 or f - c == f1 - c1: ch_antes += 1
        if f + c == f2 + c2 or f - c == f2 - c2: ch_antes += 1
        if f + c == f3 + c3 or f - c == f3 - c3: ch_antes += 1
        
    ch_despues = 0
    for c in range(n):
        if c == c1 or c == c2 or c == c3: continue
        f = tablero[c]
        if f + c == f3 + c1 or f - c == f3 - c1: ch_despues += 1
        if f + c == f1 + c2 or f - c == f1 - c2: ch_despues += 1
        if f + c == f2 + c3 or f - c == f2 - c3: ch_despues += 1
        
    return ch_antes - ch_despues


# =================================================================
# 3. WORKER DE REPARACIÓN LOCAL (GLS + VNS + TABÚ)
# =================================================================

def reparar_gls_vns_acelerado(args):
    tablero_list, n, reinas_fijas, cols_libres, max_pasos = args
    tablero = np.array(tablero_list, dtype=np.int32)
    
    pen_d1 = np.zeros(2 * n, dtype=np.float32)
    pen_d2 = np.zeros(2 * n, dtype=np.float32)
    lambda_pen = 0.5
    tabu_list = []

    for paso in range(max_pasos):
        d1_vistas, d2_vistas = {}, {}
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
        mejor_mejora = 0.0
        
        muestra_c2 = random.sample(cols_libres, min(30, len(cols_libres)))
        
        for c2 in muestra_c2:
            if c1 == c2 or (c1, c2) in tabu_list or (c2, c1) in tabu_list: continue
            mejora = evaluar_intercambio_gls_jit(tablero, c1, c2, pen_d1, pen_d2, lambda_pen)
            if mejora > mejor_mejora:
                mejor_mejora = mejora
                mejor_c2 = c2
                
        if mejor_c2 != -1:
            tablero[c1], tablero[mejor_c2] = tablero[mejor_c2], tablero[c1]
            tabu_list.append((c1, mejor_c2))
            if len(tabu_list) > 10: tabu_list.pop(0)
        else:
            conf_actuales = int(calcular_aptitud_jit(tablero))
            if conf_actuales <= 20 and len(cols_libres) >= 3:
                c2, c3 = random.sample(cols_libres, 2)
                if c1 != c2 and c1 != c3:
                    red_3swap = evaluar_3swap_jit(tablero, c1, c2, c3)
                    if red_3swap > 0:
                        f1, f2, f3 = tablero[c1], tablero[c2], tablero[c3]
                        tablero[c1], tablero[c2], tablero[c3] = f3, f1, f2
                        continue

            f1 = tablero[c1]
            pen_d1[f1 + c1] += 1.0
            pen_d2[f1 - c1 + n] += 1.0

    return tablero.tolist(), int(calcular_aptitud_jit(tablero))


# =================================================================
# 4. CLASE PRINCIPAL MEMÉTICA
# =================================================================

class MemeticoExtremoAC3:
    def __init__(self, n, reinas_fijas, tam_poblacion=20):
        self.n = n
        self.reinas_fijas = reinas_fijas
        self.tam_poblacion = tam_poblacion
        
        print(f"[*] Ejecutando preprocesamiento AC-3 sobre N={n}...")
        t_ac3 = time.perf_counter()
        self.dominios_ac3, self.cols_libres, self.d1_bloqueadas, self.d2_bloqueadas = preprocesar_dominios_ac3(n, reinas_fijas)
        print(f"[*] Poda AC-3 completada en {time.perf_counter() - t_ac3:.4f} s.")
        
        self.filas_fijas = set(reinas_fijas.values())
        self.filas_libres = [f for f in range(n) if f not in self.filas_fijas]

    def crear_individuo(self):
        return generar_individuo_ac3_mrv(
            self.n, self.reinas_fijas, self.dominios_ac3, self.cols_libres, self.d1_bloqueadas, self.d2_bloqueadas
        )

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
        print(f"[*] Motores GLS+VNS+JIT activados en {hilos} núcleos...")
        
        _ = calcular_aptitud_jit(np.zeros(self.n, dtype=np.int32))
        _ = evaluar_intercambio_gls_jit(np.zeros(self.n, dtype=np.int32), 0, 1, np.zeros(2*self.n, dtype=np.float32), np.zeros(2*self.n, dtype=np.float32), 0.5)
        _ = evaluar_3swap_jit(np.zeros(self.n, dtype=np.int32), 0, 1, 2)

        t_inicio = time.perf_counter()

        with concurrent.futures.ProcessPoolExecutor(max_workers=hilos) as executor:
            args_ini = [(self.crear_individuo(), self.n, self.reinas_fijas, self.cols_libres, 80) for _ in range(self.tam_poblacion)]
            resultados = list(executor.map(reparar_gls_vns_acelerado, args_ini))
            
            poblacion = [r[0] for r in resultados]
            aptitudes = [r[1] for r in resultados]
            
            mejor_hist = min(aptitudes)
            gens_sin_mejora = 0
            generacion = 0
            
            while True:
                mejor_actual = min(aptitudes)
                if generacion % 5 == 0:
                    print(f"  Gen {generacion:05d} -> Conflictos: {mejor_actual}")
                    
                if mejor_actual == 0:
                    return True, poblacion[aptitudes.index(0)], generacion, time.perf_counter() - t_inicio
                    
                if mejor_actual >= mejor_hist:
                    gens_sin_mejora += 1
                else:
                    mejor_hist = mejor_actual
                    gens_sin_mejora = 0
                    
                if gens_sin_mejora >= 15:
                    print(f"  🔄 REINICIO GLS (Gen {generacion}): Regenerando estocásticamente...")
                    indices = sorted(range(len(aptitudes)), key=lambda i: aptitudes[i])
                    nueva_pob = [poblacion[indices[0]], poblacion[indices[1]]]
                    
                    args_restart = [(self.crear_individuo(), self.n, self.reinas_fijas, self.cols_libres, 80) for _ in range(self.tam_poblacion - 2)]
                    resultados_restart = list(executor.map(reparar_gls_vns_acelerado, args_restart))
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
                    
                resultados_hijos = list(executor.map(reparar_gls_vns_acelerado, args_cruce))
                for r in resultados_hijos: nueva_poblacion.append(r[0])
                    
                poblacion = nueva_poblacion
                aptitudes = [mejor_actual, mejor_actual] + [r[1] for r in resultados_hijos]
                generacion += 1


# =================================================================
# 5. MAIN BENCHMARK
# =================================================================

def generar_reinas_fijas_validas(n, cantidad):
    reinas_fijas = {}
    filas_ocupadas, d1_ocupadas, d2_ocupadas = set(), set(), set()
    intentos = 0
    while len(reinas_fijas) < cantidad and intentos < cantidad * 1000:
        c, f = random.randint(0, n - 1), random.randint(0, n - 1)
        if c not in reinas_fijas and f not in filas_ocupadas:
            if (f + c) not in d1_ocupadas and (f - c) not in d2_ocupadas:
                reinas_fijas[c] = f
                filas_ocupadas.add(f)
                d1_ocupadas.add(f + c)
                d2_ocupadas.add(f - c)
        intentos += 1
    if len(reinas_fijas) < cantidad:
        return generar_reinas_fijas_validas(n, cantidad)
    return reinas_fijas


if __name__ == "__main__":
    N = 10000
    CANTIDAD_FIJAS = 4000
    
    print("==========================================================")
    print(f" PRUEBA HIPERESCALA CON AC-3 + MRV: N = {N}, K = {CANTIDAD_FIJAS}")
    print("==========================================================\n")
    
    print(f"--- GENERANDO {CANTIDAD_FIJAS} REINAS FIJAS ALEATORIAS ---")
    reinas_iniciales = generar_reinas_fijas_validas(N, CANTIDAD_FIJAS)
    print("[*] Reinas fijas colocadas. Lanzando motor...")
    
    motor = MemeticoExtremoAC3(N, reinas_iniciales, tam_poblacion=20)
    exito, _, gens, t_ejecucion = motor.resolver()

    if exito:
        print(f"\n¡RETO SUPERADO! N={N} con K={CANTIDAD_FIJAS} resuelto en {gens} generaciones.")
    print(f"Tiempo total: {t_ejecucion:.4f} segundos ({t_ejecucion / 60:.2f} minutos)")