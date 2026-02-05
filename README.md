# Wavelet-based diffusion transformer for image dehazing

Cheng Ma, Guojun Liu, Jing Yue, Wavelet-based diffusion transformer for image dehazing, Pattern Recognition Letters, Volume 201, 2026, Pages 58-65, ISSN 0167-8655, https://doi.org/10.1016/j.patrec.2026.01.016.


Abstract
In current image dehazing methods based on diffusion models, few studies explore and leverage the inherent prior knowledge of hazy images. Additionally, the inherent complexity of these models often results in difficulties during training, which in turn lead to poor restoration performance in dense hazy environments. To address these challenges, this paper proposes a dehazing diffusion model based on Haar wavelet priors, aiming to fully exploit the characteristic that haze information is concentrated in the low-frequency region. Specifically, the Haar wavelet transform is first applied to decompose the hazy image, and the diffusion model is used to generate low-frequency information in the image, thereby reconstructing the main colors and content of the dehazed image. Moreover, a high-frequency enhancement module based on Gabor is designed to extract high-frequency details through multi-directional Gabor convolution filters, further improving the fine-grained restoration capability of the image. Subsequently, a multi-scale pooling block is adopted to reduce blocky artifacts caused by non-uniform haze conditions, enhancing the visual consistency of the image. Finally, the effectiveness of the proposed method is demonstrated on publicly available datasets, and the model’s generalization ability is tested on real hazy image datasets, as well as its potential for application in other downstream tasks. 


<img width="960" height="413" alt="image" src="https://github.com/user-attachments/assets/b8a8db9e-819e-4aef-a2e9-1392b3e8178c" />


<img width="926" height="504" alt="image" src="https://github.com/user-attachments/assets/285b547b-f069-406c-9f16-fbe814e6ba37" />


<img width="948" height="509" alt="image" src="https://github.com/user-attachments/assets/cb461d50-62ad-4001-a071-d642573cfe67" />
