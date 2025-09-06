from asset.stroke_vector_mapping import *
from tools.tool import *
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    print("path: " + os.getcwd())
    for i in range(0, 40):
        for j in range(0, 21):
            if j <= 19:
                continue
            mask_key = np.zeros((500, 500))
            masks = [np.zeros((500, 500), dtype=bool) for _ in range(5)]
            for k in range(1, STROKE_VECTOR_MAP[i][0] + 1):
                img_path = f'data/output_img/{i}/{j}/{k}.jpg'
                mask_key = np.logical_or(mask_key, extract_green_mask_hsv(img_path))
                masks[STROKE_VECTOR_MAP[i][k] - 1] = np.logical_or(masks[STROKE_VECTOR_MAP[i][k] - 1],
                                                                   create_and_visualize_mask(img_path))
            save_path = f'data/output_img/{i}/{j}/mask_key_point.npy'
            np.save(save_path, mask_key)

            masks.append(mask_key)
            mask_500x500x6 = np.stack(masks, axis=-1)
            save_path = f'data/output_img/{i}/{j}/0.npy'
            np.save(save_path, mask_500x500x6)
            masks.pop()

            for idx, mask in enumerate(masks, start=1):
                save_path = f'data/output_img/{i}/{j}/mask_{idx}.npy'
                np.save(save_path, mask)

            mask_key_color = np.zeros((500, 500, 3), dtype=np.uint8)
            mask_key_color[mask_key == 1] = [0, 255, 0]

            base_img_path = f'data/output_img/{i}/{j}/0.jpg'
            base_img = Image.open(base_img_path).convert('RGB').resize((500, 500))
            base_img_array = np.array(base_img)

            overlay_img = cv2.addWeighted(base_img_array, 1.0, mask_key_color, 0.6, 0)

            save_path = f'data/output_img/{i}/{j}/key_point.png'
            Image.fromarray(overlay_img).save(save_path)

            fig, axes = plt.subplots(1, len(masks), figsize=(20, 5))
            for idx, mask in enumerate(masks):
                axes[idx].imshow(mask, cmap='gray')
                axes[idx].axis('off')

            save_path = f'data/output_img/{i}/{j}/mask_img.jpg'
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
