import os
import struct
import argparse
import numpy as np

# Класс для хранения параметров изображения COLMAP
class ColmapImage:
    def __init__(self, id, qvec, tvec, camera_id, name, xys, point3D_ids):
        self.id = id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = name
        self.xys = xys
        self.point3D_ids = point3D_ids


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    if not data:
        raise EOFError("Неожиданный конец файла при чтении.")
    return struct.unpack(endian_character + format_char_sequence, data)


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    if isinstance(data, np.ndarray):
        data = data.tolist()
    if isinstance(data, (list, tuple)):
        packed_data = struct.pack(endian_character + format_char_sequence, *data)
    else:
        packed_data = struct.pack(endian_character + format_char_sequence, data)
    fid.write(packed_data)


def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        # Считываем количество изображений (uint64)
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            # Считываем основные свойства (64 байта)
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            
            # Считываем имя файла (символ за символом до нулевого байта)
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
                
            # Считываем количество 2D точек
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            
            # Считываем сами 2D точки (каждая точка: double, double, int64 = 24 байта)
            if num_points2D > 0:
                x_y_id_s = read_next_bytes(
                    fid, num_bytes=24 * num_points2D, format_char_sequence="ddq" * num_points2D
                )
                xys = np.column_stack([
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3]))
                ])
                point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            else:
                xys = np.empty((0, 2), dtype=np.float64)
                point3D_ids = np.empty((0,), dtype=np.int64)
                
            images[image_id] = ColmapImage(
                id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id,
                name=image_name, xys=xys, point3D_ids=point3D_ids
            )
    return images


def write_images_binary(images, path_to_model_file):
    with open(path_to_model_file, "wb") as fid:
        # Пишем общее количество изображений
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            # Имя файла с нуль-терминатором на конце
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            # Пишем 2D точки
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")


def main():
    parser = argparse.ArgumentParser(description="Равномерное сэмплирование COLMAP images.bin файла.")
    parser.add_argument("-i", "--input", required=True, help="Путь к исходному файлу images.bin")
    parser.add_argument("-o", "--output", required=True, help="Путь к создаваемому урезанному файлу images.bin")
    parser.add_argument("-n", "--num", type=int, required=True, help="Сколько изображений оставить (N)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Ошибка: Исходный файл не найден по пути: {args.input}")
        return

    # Загружаем
    try:
        images = read_images_binary(args.input)
    except Exception as e:
        print(f"Не удалось прочесть файл: {e}")
        return

    total_images = len(images)
    print(f"Успешно загружено {total_images} изображений.")

    if args.num >= total_images:
        print(f"Указанное N ({args.num}) больше или равно количеству картинок в файле ({total_images}).")
        print("Сэмплирование не требуется, копия оригинального файла сохранена.")
        write_images_binary(images, args.output)
        return

    # Сортируем ключи по имени изображения (хронологический порядок кадров)
    sorted_keys = sorted(images.keys(), key=lambda x: images[x].name)

    # Вычисляем индексы для равномерного сэмплирования
    selected_indices = np.round(np.linspace(0, total_images - 1, args.num)).astype(int)
    selected_keys = [sorted_keys[idx] for idx in selected_indices]

    # Фильтруем словарь
    filtered_images = {k: images[k] for k in selected_keys}

    # Сохраняем
    write_images_binary(filtered_images, args.output)
    print(f"Сохранено {len(filtered_images)} изображений в {args.output}")


if __name__ == "__main__":
    main()