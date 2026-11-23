import time

class MotorCSPBasico:
    def __init__(self, n, reinas_fijas):
        self.n = n
        self.tablero = [-1] * n
        self.nodos = 0
        
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
        libres = 0
        for c in range(self.n):
            if self.tablero[c] == -1:
                for f in range(self.n):
                    if self.es_valida(f, c):
                        libres += 1
        return libres

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
            self._quitar_reina(fila, mejor_col)
            opciones_ordenadas.append((libres, fila))

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

    print(f"--- EJECUTANDO MOTOR BASE N={N} ---")
    motor_base = MotorCSPBasico(N, reinas_iniciales)
    t_inicio = time.perf_counter()
    exito_base = motor_base.resolver()
    t_fin = time.perf_counter()

    print(f"Solución encontrada: {exito_base}")
    print(f"Nodos explorados: {motor_base.nodos}")
    print(f"Tiempo total: {(t_fin - t_inicio):.4f} segundos\n")