import time
import random

class AlgoritmoMemeticoMasivo:
    def __init__(self, n, reinas_fijas, tam_poblacion=20):
        # Para N=1000, reducimos la población a 20. La clave es la "calidad" de la reparación,
        # no tener miles de copias malas dando vueltas.
        self.n = n
        self.reinas_fijas = reinas_fijas
        self.tam_poblacion = tam_poblacion
        
        self.filas_fijas = set(reinas_fijas.values())
        self.cols_libres = [c for c in range(n) if c not in reinas_fijas]
        self.filas_libres = [f for f in range(n) if f not in self.filas_fijas]

    def crear_individuo_aleatorio(self):
        tablero = [-1] * self.n
        for c, f in self.reinas_fijas.items():
            tablero[c] = f
            
        filas_mezcladas = self.filas_libres[:]
        random.shuffle(filas_mezcladas)
        for c, f in zip(self.cols_libres, filas_mezcladas):
            tablero[c] = f
        return tablero

    def calcular_conflictos_diagonales(self, tablero):
        conflictos = 0
        d1_counts = {}
        d2_counts = {}
        for c in range(self.n):
            f = tablero[c]
            d1_counts[f + c] = d1_counts.get(f + c, 0) + 1
            d2_counts[f - c] = d2_counts.get(f - c, 0) + 1
            
        for count in d1_counts.values():
            if count > 1: conflictos += (count * (count - 1)) // 2
        for count in d2_counts.values():
            if count > 1: conflictos += (count * (count - 1)) // 2
        return conflictos

    def obtener_columnas_conflictivas(self, tablero):
        conflictos_col = set()
        d1_vistas = {}
        d2_vistas = {}

        for c in range(self.n):
            f = tablero[c]
            d1_vistas.setdefault(f + c, []).append(c)
            d2_vistas.setdefault(f - c, []).append(c)

        for cols in d1_vistas.values():
            if len(cols) > 1: conflictos_col.update(cols)
        for cols in d2_vistas.values():
            if len(cols) > 1: conflictos_col.update(cols)

        return [c for c in conflictos_col if c not in self.reinas_fijas]

    def reparacion_heuristica_rapida(self, tablero, max_pasos=20):
        """
        VERSIÓN OPTIMIZADA PARA N=1000:
        En lugar de recalcular todo el tablero (O(N)), calculamos solo el impacto del cambio (O(1)).
        """
        for _ in range(max_pasos):
            cols_conflicto = self.obtener_columnas_conflictivas(tablero)
            if not cols_conflicto:
                break
                
            # Cogemos una reina conflictiva al azar
            c1 = random.choice(cols_conflicto)
            f1 = tablero[c1]
            
            mejor_c2 = -1
            mejor_reduccion = 0 # Buscamos reducciones de conflictos > 0
            
            # Solo miramos 20 columnas al azar para no perder tiempo (Muestreo ligero)
            muestra_c2 = random.sample(self.cols_libres, min(20, len(self.cols_libres)))
            
            for c2 in muestra_c2:
                if c1 == c2: continue
                f2 = tablero[c2]
                
                # Cuántos choques tienen las reinas en sus posiciones ACTUALES
                choques_antes = self._contar_choques_locales(tablero, c1, f1) + \
                                self._contar_choques_locales(tablero, c2, f2)
                                
                # Cuántos choques tendrían si las INTERCAMBIAMOS
                choques_despues = self._contar_choques_locales(tablero, c1, f2, ignorar_c=c2) + \
                                  self._contar_choques_locales(tablero, c2, f1, ignorar_c=c1)
                                  
                reduccion = choques_antes - choques_despues
                
                if reduccion > mejor_reduccion:
                    mejor_reduccion = reduccion
                    mejor_c2 = c2
                    
            if mejor_c2 != -1:
                tablero[c1], tablero[mejor_c2] = tablero[mejor_c2], tablero[c1]
            else:
                # Si no encontramos mejora, forzamos un pequeño cambio aleatorio para evitar atascos locales
                if random.random() < 0.1:
                    c_rand = random.choice(self.cols_libres)
                    tablero[c1], tablero[c_rand] = tablero[c_rand], tablero[c1]

        return tablero

    def _contar_choques_locales(self, tablero, c_eval, f_eval, ignorar_c=-1):
        """Cuenta los choques de UNA sola reina sin mirar todo el tablero."""
        choques = 0
        for c in range(self.n):
            if c == c_eval or c == ignorar_c: continue
            f = tablero[c]
            if f + c == f_eval + c_eval or f - c == f_eval - c_eval:
                choques += 1
        return choques

    def cruzar_ordenado(self, padre1, padre2):
        hijo = [-1] * self.n
        for c, f in self.reinas_fijas.items():
            hijo[c] = f
            
        if not self.cols_libres: return padre1[:]
        punto_corte = random.randint(0, len(self.cols_libres) - 1)
        cols_heredadas = set(self.cols_libres[:punto_corte])
        
        filas_usadas = set()
        for c in cols_heredadas:
            hijo[c] = padre1[c]
            filas_usadas.add(padre1[c])
            
        idx_hijo = punto_corte
        for c in self.cols_libres:
            fila_p2 = padre2[c]
            if fila_p2 not in filas_usadas:
                c_hijo = self.cols_libres[idx_hijo]
                hijo[c_hijo] = fila_p2
                filas_usadas.add(fila_p2)
                idx_hijo += 1
                
        return hijo

    def resolver(self, max_generaciones=500):
        # Población inicial mejorada
        poblacion = []
        for _ in range(self.tam_poblacion):
            ind = self.crear_individuo_aleatorio()
            ind = self.reparacion_heuristica_rapida(ind, max_pasos=50) # Empuje fuerte inicial
            poblacion.append(ind)
            
        aptitudes = [self.calcular_conflictos_diagonales(ind) for ind in poblacion]
        
        for generacion in range(max_generaciones):
            mejor_aptitud = min(aptitudes)
            
            # Para N=1000, imprimimos el progreso cada 10 generaciones para ver que no está colgado
            if generacion % 10 == 0:
                print(f"Gen {generacion:03d} -> Mejor tablero actual tiene {mejor_aptitud} conflictos restantes")
                
            if mejor_aptitud == 0:
                return True, poblacion[aptitudes.index(0)], generacion
                
            nueva_poblacion = []
            
            # Elitismo: Los 2 mejores pasan directo
            indices = sorted(range(len(aptitudes)), key=lambda i: aptitudes[i])
            nueva_poblacion.append(poblacion[indices[0]])
            nueva_poblacion.append(poblacion[indices[1]])
            
            while len(nueva_poblacion) < self.tam_poblacion:
                t1 = random.sample(list(zip(poblacion, aptitudes)), 3)
                p1 = min(t1, key=lambda x: x[1])[0]
                t2 = random.sample(list(zip(poblacion, aptitudes)), 3)
                p2 = min(t2, key=lambda x: x[1])[0]
                
                hijo = self.cruzar_ordenado(p1, p2)
                hijo = self.reparacion_heuristica_rapida(hijo, max_pasos=15)
                nueva_poblacion.append(hijo)
                
            poblacion = nueva_poblacion
            aptitudes = [self.calcular_conflictos_diagonales(ind) for ind in poblacion]
            
        return False, poblacion[aptitudes.index(min(aptitudes))], max_generaciones


if __name__ == "__main__":
    N = 1000
    
    # 15 reinas fijas distribuidas proporcionalmente en el enorme tablero de 1000x1000
    filas_iniciales = [20, 50, 90, 150, 220, 270, 320, 370, 420, 470, 520, 570, 620, 670, 720]
    reinas_iniciales = {col * 10: fila for col, fila in enumerate(filas_iniciales)}

    print(f"--- EJECUTANDO ALGORITMO MEMÉTICO MASIVO N={N} ---")
    print(f"Tablero de {N*N} casillas. Población: 20.")
    print("El algoritmo imprimirá su progreso cada 10 generaciones...\n")
    
    motor_masivo = AlgoritmoMemeticoMasivo(N, reinas_iniciales, tam_poblacion=20)
    
    t_inicio = time.perf_counter()
    exito, mejor_tablero, generaciones = motor_masivo.resolver(max_generaciones=500)
    t_fin = time.perf_counter()

    if exito:
        print(f"\n¡VICTORIA! Solución PERFECTA encontrada en N={N}.")
    else:
        conflictos = motor_masivo.calcular_conflictos_diagonales(mejor_tablero)
        print(f"\nLímite alcanzado. Mejor tablero con {conflictos} conflictos restantes.")
        
    print(f"Generaciones evolucionadas: {generaciones}")
    print(f"Tiempo total: {(t_fin - t_inicio):.4f} segundos")