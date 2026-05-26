import sys
import numpy as np
from plyfile import PlyData
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Header, Footer, Static
from textual.containers import Container

class PlyViewer(App):
    """Интерактивный просмотрщик характеристик .ply файлов."""
    
    BINDINGS = [
        ("q", "quit", "Выход"),
        ("s", "sort", "Сортировать по столбцу"),
    ]
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.data_rows = []
        self.column_names = []

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        
        # Загрузка данных
        self.load_ply_data()
        
        # Добавляем колонки
        table.add_columns(*self.column_names)
        
        # Добавляем строки (ограничим 1000 для скорости)
        table.add_rows(self.data_rows[:1000])

    def load_ply_data(self):
        try:
            print(f"Читаю {self.file_path}...")
            plydata = PlyData.read(self.file_path)
            vertex = plydata['vertex']
            
            # Получаем имена всех свойств (x, y, z, opacity, f_dc_0, f_rest_0 и т.д.)
            self.column_names = [prop.name for prop in vertex.properties]
            
            # Конвертируем в список строк для таблицы
            # Берем срез, так как в 3DGS может быть 1-5 млн точек
            raw_data = vertex.data
            count = min(len(raw_data), 1000)
            
            for i in range(count):
                row = [f"{raw_data[i][name]:.4f}" if isinstance(raw_data[i][name], (float, np.float32)) 
                       else str(raw_data[i][name]) 
                       for name in self.column_names]
                self.data_rows.append(row)
                
        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")
            sys.exit(1)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(DataTable())
        yield Static(f" Файл: {self.file_path} (Показаны первые 1000 точек)", id="info")
        yield Footer()

    def action_quit(self):
        self.exit()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python ply_viewer.py <путь_к_файлу.ply>")
        sys.exit(1)
        
    path = sys.argv[1]
    app = PlyViewer(path)
    app.run()