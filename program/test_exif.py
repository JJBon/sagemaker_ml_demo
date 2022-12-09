
from PIL import Image
import piexif

folder_path = '/opt/ml/input/data/training'
img_filename = 'Black-grass/1.png'
img_path = f'{folder_path}/{img_filename}'
im = Image.open(f'{folder_path}/{img_filename}')

metadata = {
        "file_name": img_filename
    
}

exif = im.getexif()
exif[0x9286] = str(metadata)
im.save(f"/opt/ml/input/data/training_refined/{img_filename}", exif=exif)

#load exif data
exif_dict =  im.getexif()
print("default dict ", exif_dict)
#exif_bytes = piexif.dump(metadata)
#piexif.insert(exif_bytes, "/opt/ml/input/data/training_refined/cookie.jpg")
print("image saved")


# #UserComment


#im.save('/opt/ml/input/data/training_refined/cookie.jpg', exif=exif_bytes)


folder_path = '/opt/ml/input/data/training_refined'
img_path = f'{folder_path}/{img_filename}'
im = Image.open(f'{folder_path}/{img_filename}')

print("image metadata ", im.getexif())



# from exif import Image

# folder_path = '/opt/ml/input/data/training/Black-grass'
# img_filename = '1.png'
# img_path = f'{folder_path}/{img_filename}'

# with open(img_path, 'rb') as img_file:
#     img = Image(img_file)
#     metadata = {
#         "file_name": img_path
#     }
#     print(metadata)
    
#     img.user_comment = str(metadata)
    
    
# print(img.has_exif)
# print(img.user_comment)

# if img.has_exif:
#     with open(f'/opt/ml/input/data/training_refined/cookie.jpg', 'wb') as new_image_file:
#         new_image_file.write(img.get_file())