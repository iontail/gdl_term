from augmentation.new_augment import Mixer

if __name__ == "__main__":
    img_size = 32
    ratio = 0.5
    alpha = 0.2
    active_lam = False
    retain_lam = False

    check = input(f"Are you sure to create new augmented data with img_size={img_size}, ratio={ratio}, alpha={alpha}, active_lam={active_lam}, retain_lam={retain_lam}? (y/n): ")
    if check.lower().strip() == 'y':
        print("Starting augmentation process...")

        mixer = Mixer(img_size=img_size,
                      load=False,
                      ratio=ratio,
                      alpha=alpha,
                      active_lam=active_lam,
                      retain_lam=retain_lam)
        
        mixer.generate_augmented_imgs(base_directory='./datasets')
        print("Augmentation process completed.")
        
    else:
        print("Augmentation process canceled.")