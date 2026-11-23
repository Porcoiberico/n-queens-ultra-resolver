import time
import random

class AlgoritmoMemeticoNReinas:
    def __init__(self, n, reinas_fijas, tam_poblacion=50):
        # Población más pequeña (50) porque ahora cada individuo es "muy inteligente"
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
        """Detecta exactamente qué columnas están sufriendo ataques."""
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

        # Solo podemos mover las que no están fijas
        return [c for c in conflictos_col if c not in self.reinas_fijas]

    def aplicar_heuristica_local(self, tablero, max_pasos=5):
        """
        LA FUSIÓN: Tu heurística de amenazas entra en acción.
        El algoritmo ya no muta a ciegas. Prueba movimientos y elige el que minimiza las amenazas.
        """
        for _ in range(max_pasos):
            cols_conflicto = self.obtener_columnas_conflictivas(tablero)
            if not cols_conflicto:
                break  # Si ya no hay conflictos, el tablero es perfecto

            # Cogemos una reina que esté siendo atacada
            c1 = random.choice(cols_conflicto)
            
            menor_amenaza = self.calcular_conflictos_diagonales(tablero)
            mejor_c2 = -1

            # Tu idea: Muestreamos 15 posibles casillas/columnas libres para intercambiarla
            muestra_c2 = random.sample(self.cols_libres, min(15, len(self.cols_libres)))

            for c2 in muestra_c2:
                if c1 == c2: continue

                # Simulamos el intercambio
                tablero[c1], tablero[c2] = tablero[c2], tablero[c1]
                amenazas_actuales = self.calcular_conflictos_diagonales(tablero)

                # Evaluamos: ¿Se reducen las amenazas internas?
                if amenazas_actuales < menor_amenaza:
                    menor_amenaza = amenazas_actuales
                    mejor_c2 = c2

                # Deshacemos la simulación
                tablero[c1], tablero[c2] = tablero[c2], tablero[c1]

            # Aplicamos el MEJOR movimiento encontrado (Búsqueda local guiada)
            if mejor_c2 != -1:
                tablero[c1], tablero[mejor_c2] = tablero[mejor_c2], tablero[c1]
            else:
                break # Si ningún movimiento lo mejora, paramos de optimizar este tablero

        return tablero

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

    def resolver(self, max_generaciones=1000):
        # 1. Población inicial (y la optimizamos inmediatamente con tu heurística)
        poblacion = []
        for _ in range(self.tam_poblacion):
            ind = self.crear_individuo_aleatorio()
            ind = self.aplicar_heuristica_local(ind)  # ¡Magia aquí!
            poblacion.append(ind)
            
        aptitudes = [self.calcular_conflictos_diagonales(ind) for ind in poblacion]
        
        for generacion in range(max_generaciones):
            mejor_aptitud_actual = min(aptitudes)
            if mejor_aptitud_actual == 0:
                return True, poblacion[aptitudes.index(0)], generacion
                
            nueva_poblacion = []
            
            # Elitismo: Guardamos al mejor
            indices_ordenados = sorted(range(len(aptitudes)), key=lambda i: aptitudes[i])
            nueva_poblacion.append(poblacion[indices_ordenados[0]])
            
            while len(nueva_poblacion) < self.tam_poblacion:
                # Selección
                torneo1 = random.sample(list(zip(poblacion, aptitudes)), 3)
                padre1 = min(torneo1, key=lambda x: x[1])[0]
                torneo2 = random.sample(list(zip(poblacion, aptitudes)), 3)
                padre2 = min(torneo2, key=lambda x: x[1])[0]
                
                # Cruce genético
                hijo = self.cruzar_ordenado(padre1, padre2)
                
                # Mutación Aleatoria muy baja (5%) solo para evitar atascos absolutos
                if random.random() < 0.05:
                    c1, c2 = random.sample(self.cols_libres, 2)
                    hijo[c1], hijo[c2] = hijo[c2], hijo[c1]

                # EL TOQUE MAESTRO: Reparación guiada por heurística de amenazas
                hijo = self.aplicar_heuristica_local(hijo)
                
                nueva_poblacion.append(hijo)
                
            poblacion = nueva_poblacion
            aptitudes = [self.calcular_conflictos_diagonales(ind) for ind in poblacion]
            
        return False, poblacion[aptitudes.index(min(aptitudes))], max_generaciones

# =====================================================================
if __name__ == "__main__":
    N = 100
    filas_iniciales = [2, 5, 9, 15, 22, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72]
    reinas_iniciales = {col: fila for col, fila in zip(range(len(filas_iniciales)), filas_iniciales)}

    print(f"--- EJECUTANDO ALGORITMO HÍBRIDO (MEMÉTICO) N={N} ---")
    print("Combinando búsqueda global con Heurística de Reducción de Amenazas...")
    
    motor_hibrido = AlgoritmoMemeticoNReinas(N, reinas_iniciales, tam_poblacion=50)
    
    t_inicio = time.perf_counter()
    exito, mejor_tablero, generaciones = motor_hibrido.resolver()
    t_fin = time.perf_counter()

    if exito:
        print("\n¡Solución PERFECTA (0 conflictos) encontrada!")
    else:
        conflictos = motor_hibrido.calcular_conflictos_diagonales(mejor_tablero)
        print(f"\nLímite alcanzado. Mejor tablero con {conflictos} conflictos restantes.")
        
    print(f"Generaciones evolucionadas: {generaciones}")
    print(f"Tiempo total: {(t_fin - t_inicio):.4f} segundos")