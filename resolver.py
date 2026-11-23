import time

class MotorCSPMejorado:
    def __init__(self, n, reinas_fijas, peso_amenazas=1.5):
        self.n = n
        self.tablero = [-1] * n
        self.nodos = 0
        self.peso_amenazas = peso_amenazas
        
        self.filas_ocupadas = set()
        self.diag_asc = set()
        self.diag_desc = set()
        
        for c, f in reinas_fijas.items():
            self._colocar_reina(f, c)

    def _colocar_reina(self, f, c):
        self.tablero[c] = f
        self.filas_ocupadas.add(f)
        self.diag_asc.add(f + c)
        self.diag_desc.add(f - c)

    def _quitar_reina(self, f, c):
        self.tablero[c] = -1
        self.filas_ocupadas.remove(f)
        self.diag_asc.remove(f + c)
        self.diag_desc.remove(f - c)

    def es_valida(self, f, c):
        return (f not in self.filas_ocupadas and 
                (f + c) not in self.diag_asc and 
                (f - c) not in self.diag_desc)

    def obtener_opciones_columna(self, c):
        return [f for f in range(self.n) if self.es_valida(f, c)]

    def obtener_casillas_libres_totales(self):
        libres = []
        for c in range(self.n):
            if self.tablero[c] == -1:
                for f in range(self.n):
                    if self.es_valida(f, c):
                        libres.append((f, c))
        return libres

    def contar_amenazas_internas_muestreo(self, casillas_libres):
        if not casillas_libres:
            return 0
            
        min_fila = min(f for f, c in casillas_libres)
        
        # MUESTREO DINÁMICO: El radio se ajusta automáticamente al 10% de N
        radio = max(2, int(self.n * 0.10))
        capa_cercana = [pt for pt in casillas_libres if pt[0] <= min_fila + radio]
        
        amenazas = 0
        k = len(capa_cercana)
        for i in range(k):
            f1, c1 = capa_cercana[i]
            for j in range(i + 1, k):
                f2, c2 = capa_cercana[j]
                if f1 == f2 or c1 == c2 or abs(f1 - f2) == abs(c1 - c2):
                    amenazas += 1
        return amenazas

    def resolver(self):
        self.nodos += 1

        mejor_col = -1
        min_opciones = float('inf')
        
        for c in range(self.n):
            if self.tablero[c] == -1:
                opciones = self.obtener_opciones_columna(c)
                if len(opciones) < min_opciones:
                    min_opciones = len(opciones)
                    mejor_col = c

        if mejor_col == -1:
            return True
        
        if min_opciones == 0:
            return False

        filas_candidatas = self.obtener_opciones_columna(mejor_col)
        
        opciones_ordenadas = []
        for fila in filas_candidatas:
            self._colocar_reina(fila, mejor_col)
            
            libres = self.obtener_casillas_libres_totales()
            amenazas = self.contar_amenazas_internas_muestreo(libres)
            
            puntuacion = len(libres) - (self.peso_amenazas * amenazas)
            
            self._quitar_reina(fila, mejor_col)
            
            opciones_ordenadas.append((puntuacion, fila))

        opciones_ordenadas.sort(key=lambda x: x[0], reverse=True)

        for _, fila in opciones_ordenadas:
            self._colocar_reina(fila, mejor_col)
            if self.resolver():
                return True
            self._quitar_reina(fila, mejor_col)

        return False

if __name__ == "__main__":
    N = 100
    filas_iniciales = [2, 5, 9, 15, 22, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72]
    reinas_iniciales = {col: fila for col, fila in zip(range(len(filas_iniciales)), filas_iniciales)}

    print(f"--- EJECUTANDO MOTOR MEJORADO N={N} ---")
    motor_mejorado = MotorCSPMejorado(N, reinas_iniciales, peso_amenazas=1.5)
    t_inicio = time.perf_counter()
    exito_m = motor_mejorado.resolver()
    t_fin = time.perf_counter()

    print(f"Solución encontrada: {exito_m}")
    print(f"Nodos explorados: {motor_mejorado.nodos}")
    print(f"Tiempo total: {(t_fin - t_inicio):.4f} segundos")