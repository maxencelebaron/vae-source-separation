import torch
import torch.optim as optim
from torch import nn


class VAE_Separator:
    def __init__(self, vae_s1, vae_s2, device):
        self.vae_s1 = vae_s1  # Speech Model
        self.vae_s2 = vae_s2  # Noise Model
        self.device = device

        # On freeze les VAEs (il ne s'agit plus d'un entrainement)
        self.vae_s1.eval()
        self.vae_s2.eval()
        for p in self.vae_s1.parameters(): p.requires_grad = False
        for p in self.vae_s2.parameters(): p.requires_grad = False

    def separate(self, mix_log_mag, num_steps=100, lr=0.1):
        """
        mix_magnitude: [B, 1, F, T]
        """
        B, C, F, T = mix_log_mag.shape
        latent_dim = self.vae_s1.encoder.dim_z

        # Initialisation aléatoire des vecteurs de l'espace latent
        # Le but est de les optimiser pour retrouver les signaux individuels
        z1 = torch.randn(B, T, latent_dim, device=self.device, requires_grad=True)
        z2 = torch.randn(B, T, latent_dim, device=self.device, requires_grad=True)

        optimizer = optim.Adam([z1, z2], lr=lr)
        loss_fn = nn.MSELoss()

        # Optimization
        # On veut trouver z1 et z2 tels que Decoder(z1) + Decoder(z2) == Mixture
        for i in range(num_steps):
            optimizer.zero_grad()

            # Passage de z dans les Decoders
            log_est_s1 = self.vae_s1.decoder(z1)
            log_est_s2 = self.vae_s2.decoder(z2)

            # Conversion échelle linéaire
            lin_est_s1 = torch.expm1(log_est_s1)
            lin_est_s2 = torch.expm1(log_est_s2)

            # On assure la positivité
            lin_est_s1 = torch.relu(lin_est_s1) + 1e-8
            lin_est_s2 = torch.relu(lin_est_s2) + 1e-8

            # Reconstructiond de la Mixture
            lin_mix_est = lin_est_s1 + lin_est_s2

            log_mix_est = torch.log1p(lin_mix_est)

            # Calcul de la loss
            loss = loss_fn(log_mix_est, mix_log_mag)

            loss.backward()
            optimizer.step()

        # On renvoie les amplitudes
        with torch.no_grad():
            final_s1 = self.vae_s1.decoder(z1).detach()
            final_s2 = self.vae_s2.decoder(z2).detach()

        return final_s1, final_s2


def apply_wiener_filter(mix_log_mag, log_est_s1, log_est_s2, eps=1e-8):
    """
    Prend les entrées en échelle log et renvoie en linéaire
    """
    # Conversion en échelle linéaire
    mix_lin = torch.expm1(mix_log_mag)
    s1_lin = torch.expm1(log_est_s1)
    s2_lin = torch.expm1(log_est_s2)

    # Calcul du masque
    mask_s1 = s1_lin / (s1_lin + s2_lin + eps)
    mask_s2 = s2_lin / (s1_lin + s2_lin + eps)

    # Application du masque
    sep_s1_lin = mask_s1 * mix_lin
    sep_s2_lin = mask_s2 * mix_lin

    return sep_s1_lin, sep_s2_lin


class ITL_Separator:
    """
    Séparateur pour deux modèles ITL-AE. Le principe est identique à la
    séparation par VAE
    """
    def __init__(self, itl_s1, itl_s2, device):
        self.itl_s1 = itl_s1  # Modèle ITL-AE pour la source 1
        self.itl_s2 = itl_s2  # Modèle ITL-AE pour la source 2
        self.device = device

        # On passe les modèles en mode évaluation et on gèle les paramètres
        self.itl_s1.eval()
        self.itl_s2.eval()
        for p in self.itl_s1.parameters():
            p.requires_grad = False
        for p in self.itl_s2.parameters():
            p.requires_grad = False

        # Récupération de lambda
        # On utilise le même lambda que pour la régularisation de la divergence
        self.lambda_cycle = self.itl_s1.regul_param

        print("\n[ITL_Separator] Modèles chargés et gelés.")
        print(f"[ITL_Separator] Cycle Loss Scale (λ_cycle) set to: {self.lambda_cycle}\n")

    def separate(self, mix_log_mag, num_steps=100, lr=0.1):
        """
        Séparation par optimisation des variables latentes.
        """
        B, C, F, T = mix_log_mag.shape
        latent_dim = self.itl_s1.encoder.dim_z

        # Initialisation des variables latentes Z1 et Z2
        z1 = torch.randn(B, T, latent_dim, device=self.device, requires_grad=True)
        z2 = torch.randn(B, T, latent_dim, device=self.device, requires_grad=True)

        optimizer = optim.Adam([z1, z2], lr=lr)
        mse = nn.MSELoss()

        # Boucle d’optimisation des variables latentes
        for step in range(num_steps):

            optimizer.zero_grad()

            # DÉCODAGE
            log_s1_hat = self.itl_s1.decoder(z1)    # Log magnitude estimée S1
            log_s2_hat = self.itl_s2.decoder(z2)    # Log magnitude estimée S2

            lin_s1_hat = torch.relu(torch.expm1(log_s1_hat))
            lin_s2_hat = torch.relu(torch.expm1(log_s2_hat))

            # Reconstruction de la magnitude du mélange
            lin_mix_hat = lin_s1_hat + lin_s2_hat
            log_mix_hat = torch.log1p(lin_mix_hat + 1e-8)

            #  2. PERTE DE RECONSTRUCTION DU MÉLANGE (Mix Loss)
            reconstruction_loss = mse(log_mix_hat, mix_log_mag)

            # 3. PERTE DE BOUCLE
            z1_cycle = self.itl_s1.encoder(log_s1_hat)
            z2_cycle = self.itl_s2.encoder(log_s2_hat)

            # La perte de cycle est la distance entre le latent optimisé (z) et le latent ré-encodé (z_cycle)
            cycle_loss = mse(z1, z1_cycle) + mse(z2, z2_cycle)

            # 4. PERTE TOTALE
            total_loss = reconstruction_loss + self.lambda_cycle * cycle_loss

            # Backpropagation : mise à jour de Z1, Z2
            total_loss.backward()
            optimizer.step()

        # Reconstructions finales sans gradient
        with torch.no_grad():
            final_s1 = self.itl_s1.decoder(z1).detach()
            final_s2 = self.itl_s2.decoder(z2).detach()

        return final_s1, final_s2
