import time
import random
import os
import concurrent.futures

# =================================================================
# FUNCIONES PURAS GLOBALES PARA MULTIPROCESSING
# =================================================================
def calcular_aptitud(tablero, n):
    conflictos = 0
    d1, d2 = {}, {}
    for c in range(n):
        f = tablero[c]
        d1[f + c] = d1.get(f + c, 0) + 1
        d2[f - c] = d2.get(f - c, 0) + 1
    for count in d1.values():
        if count > 1: conflictos += (count * (count - 1)) // 2
    for count in d2.values():
        if count > 1: conflictos += (count * (count - 1)) // 2
    return conflictos

def _contar_choques(tablero, n, c_eval, f_eval, ignorar_c):
    choques = 0
    for c in range(n):
        if c == c_eval or c == ignorar_c: continue
        f = tablero[c]
        if f + c == f_eval + c_eval or f - c == f_eval - c_eval:
            choques += 1
    return choques

def reparar_y_evaluar(args):
    tablero, n, reinas_fijas, cols_libres, max_pasos = args
    tabu_list = [] # LISTA TABÚ
    
    for _ in range(max_pasos):
        d1_vistas, d2_vistas = {}, {}
        conflictos_col = set()
        for c in range(n):
            f = tablero[c]
            d1_vistas.setdefault(f + c, []).append(c)
            d2_vistas.setdefault(f - c, []).append(c)
            
        for cols in d1_vistas.values():
            if len(cols) > 1: conflictos_col.update(cols)
        for cols in d2_vistas.values():
            if len(cols) > 1: conflictos_col.update(cols)
            
        cols_conf = [c for c in conflictos_col if c not in reinas_fijas]
        if not cols_conf: break
        
        c1 = random.choice(cols_conf)
        f1 = tablero[c1]
        mejor_c2 = -1
        mejor_reduccion = 0
        
        muestra_c2 = random.sample(cols_libres, min(20, len(cols_libres)))
        
        for c2 in muestra_c2:
            if c1 == c2: continue
            
            # PROTECCIÓN TABÚ: No rehacer un movimiento reciente
            if (c1, c2) in tabu_list or (c2, c1) in tabu_list:
                continue
                
            f2 = tablero[c2]
            ch_antes = _contar_choques(tablero, n, c1, f1, -1) + _contar_choques(tablero, n, c2, f2, -1)
            ch_despues = _contar_choques(tablero, n, c1, f2, c2) + _contar_choques(tablero, n, c2, f1, c1)
            
            reduccion = ch_antes - ch_despues
            if reduccion > mejor_reduccion:
                mejor_reduccion = reduccion
                mejor_c2 = c2
                
        if mejor_c2 != -1:
            tablero[c1], tablero[mejor_c2] = tablero[mejor_c2], tablero[c1]
            # Añadir a Tabú (memoria de las últimas 5 operaciones)
            tabu_list.append((c1, mejor_c2))
            if len(tabu_list) > 5: tabu_list.pop(0)
        else:
            # Mutación menor forzada si está muy atascado
            if random.random() < 0.1:
                cr = random.choice(cols_libres)
                tablero[c1], tablero[cr] = tablero[cr], tablero[c1]
                
    return tablero, calcular_aptitud(tablero, n)

# =================================================================
# CLASE PRINCIPAL
# =================================================================
class MemeticoSupremo:
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
        # Repartimos el trabajo entre TODOS los núcleos de tu CPU
        hilos = os.cpu_count() or 4
        print(f"[*] Iniciando Multiprocesamiento con {hilos} núcleos de CPU...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=hilos) as executor:
            args_ini = [(self.crear_ind(), self.n, self.reinas_fijas, self.cols_libres, 60) for _ in range(self.tam_poblacion)]
            resultados = list(executor.map(reparar_y_evaluar, args_ini))
            
            poblacion = [r[0] for r in resultados]
            aptitudes = [r[1] for r in resultados]
            
            mejor_hist = min(aptitudes)
            gens_sin_mejora = 0
            
            for generacion in range(500):
                mejor_actual = min(aptitudes)
                if generacion % 5 == 0:
                    print(f"Gen {generacion:03d} -> Conflictos: {mejor_actual}")
                    
                if mejor_actual == 0:
                    return True, poblacion[aptitudes.index(0)], generacion
                    
                # CONTROL DE REINICIOS (RESTART STRATEGY)
                if mejor_actual >= mejor_hist:
                    gens_sin_mejora += 1
                else:
                    mejor_hist = mejor_actual
                    gens_sin_mejora = 0
                    
                if gens_sin_mejora >= 10:
                    print(f"🔄 REINICIO TÁCTICO: Atascado {gens_sin_mejora} ciclos en {mejor_actual}. Regenerando población...")
                    indices = sorted(range(len(aptitudes)), key=lambda i: aptitudes[i])
                    nueva_pob = [poblacion[indices[0]], poblacion[indices[1]]] # Conservar élite
                    
                    args_restart = [(self.crear_ind(), self.n, self.reinas_fijas, self.cols_libres, 60) for _ in range(self.tam_poblacion - 2)]
                    resultados_restart = list(executor.map(reparar_y_evaluar, args_restart))
                    nueva_pob.extend([r[0] for r in resultados_restart])
                    
                    poblacion = nueva_pob
                    aptitudes = [r[1] for r in resultados_restart]
                    aptitudes.insert(0, mejor_actual)
                    aptitudes.insert(0, mejor_actual)
                    gens_sin_mejora = 0
                    continue
                    
                # CRUCE Y REPARACIÓN PARALELA
                indices = sorted(range(len(aptitudes)), key=lambda i: aptitudes[i])
                nueva_poblacion = [poblacion[indices[0]], poblacion[indices[1]]]
                
                args_cruce = []
                while len(nueva_poblacion) + len(args_cruce) < self.tam_poblacion:
                    t1 = random.sample(list(zip(poblacion, aptitudes)), 3)
                    p1 = min(t1, key=lambda x: x[1])[0]
                    t2 = random.sample(list(zip(poblacion, aptitudes)), 3)
                    p2 = min(t2, key=lambda x: x[1])[0]
                    
                    hijo = self.cruzar(p1, p2)
                    args_cruce.append((hijo, self.n, self.reinas_fijas, self.cols_libres, 20))
                    
                resultados_hijos = list(executor.map(reparar_y_evaluar, args_cruce))
                
                for r in resultados_hijos:
                    nueva_poblacion.append(r[0])
                    
                poblacion = nueva_poblacion
                # Solo evaluamos los hijos nuevos
                aptitudes = [mejor_actual, mejor_actual] + [r[1] for r in resultados_hijos]
                
        return False, poblacion[0], 500


if __name__ == "__main__":
    N = 1000
    filas_iniciales = [20, 50, 90, 150, 220, 270, 320, 370, 420, 470, 520, 570, 620, 670, 720]
    reinas_iniciales = {col * 10: fila for col, fila in enumerate(filas_iniciales)}

    print(f"--- EJECUTANDO ALGORITMO MEMÉTICO SUPREMO N={N} ---")
    motor = MemeticoSupremo(N, reinas_iniciales, tam_poblacion=20)
    
    t_inicio = time.perf_counter()
    exito, _, gens = motor.resolver()
    t_fin = time.perf_counter()

    if exito:
        print(f"\n¡VICTORIA MASIVA! Solución PERFECTA encontrada en {gens} generaciones.")
    print(f"Tiempo total: {(t_fin - t_inicio):.4f} segundos")