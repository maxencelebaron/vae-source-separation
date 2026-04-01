import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class Encoder(nn.Module):
    '''
    Prend le spectrogramme x et renvoie les paramètres mu(phi), v(phi) et la variable latente z
    '''
    def __init__(self,
                 input_dim: int = 257,  # longueur dimension frequentielle
                 latent_dim: int = 32):  # dimension de l'espace latent
        super(Encoder, self).__init__()

        self.dim_z = latent_dim
        hidden_dim = 128  # dimension couche cachée du MLP
        output_dim = 2 * latent_dim
        hidden_lstm_dim = 128

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_lstm_dim,
            num_layers=1,
            batch_first=True,  # [B,T,F]
            bidirectional=True
        )

        listm_output_dim = 2*hidden_lstm_dim

        # MLP:
        self.mlp = nn.Sequential(
            nn.Linear(listm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x est [B,1,F,T]
        B, C, F, T = x.shape
        x_reshape = x.squeeze().permute(0, 2, 1) # [B,T,F]
        # x_flat = x.permute(0, 3, 1, 2).reshape(B*T,F) # [B*T,F]

        # LSTM
        lstm_output, _ = self.lstm(x_reshape) # [B,T, 2*DIM_LSTM]
        # MLP
        raw_output = self.mlp(lstm_output).reshape(B*T, 2*self.dim_z) # [B*T, 2*DIM_LSTM]

        mu_phi_flat = raw_output[:, :self.dim_z] # [B*T, DIM_Z]
        v_phi_flat = raw_output[:, self.dim_z:] # [B*T, DIM_Z]

        # reparametrisation trick
        std_phi_flat = torch.exp(0.5*v_phi_flat) # [B*T, Dim_z]
        eps = torch.randn_like(std_phi_flat) # [B*T, Dim_z]
        z_flat = mu_phi_flat + std_phi_flat * eps # [B*T, Dim_z]

        mu_phi = mu_phi_flat.reshape(B, T, self.dim_z) # [B,T,Dim_z]
        v_phi = v_phi_flat.reshape(B, T, self.dim_z) # [B,T,Dim_z]
        z = z_flat.reshape(B, T, self.dim_z) # [B,T,Dim_z]

        return mu_phi, v_phi, z


class Decoder(nn.Module):
    '''
    Reconstruit le signal à partir de z.
    '''
    def __init__(self, latent_dim: int = 32, # dimension de l'espace latent
                 output_dim: int = 257): # longueur dimension frequentielle
        super(Decoder, self).__init__()

        # dimension de sortie
        self.output_dim = output_dim

        # Entrée du MLP : dimension latente (DIM_Z)
        hidden_dim = 128 # Dimension cachée du MLP
        hidden_lstm_dim = 128 # Dimension cachée du LSTM
        # Dimension de sortie : N_FREQ_BINS

        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_lstm_dim,
            num_layers=1,
            batch_first=True, # [B,T,F]
            bidirectional=True
        )

        lstm_output_dim = 2 * hidden_lstm_dim

        # MLP qui ressemble à l'encodeur
        self.mlp = nn.Sequential(
            # Input: [B*T, DIM_Z]
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) # Sortie : [B*T, N_FREQ_BINS]
        )

    def forward(self, z):
        # Entrée: z de dimension : [B, T, Dim_z]
        B, T, D = z.shape

        # LSTM
        lstm_out, _ = self.lstm(z)

        # Flatten pour le MLP
        # z_flat = z.reshape(B * T, D) # [B * T, Dim_z]

        # MLP
        raw_output = self.mlp(lstm_out) # [B * T, N_FREQ_BINS]
        X_hat_BTF = raw_output.reshape(B, T, self.output_dim)  # [B, T, F]
        X_hat = X_hat_BTF.permute(0, 2, 1).unsqueeze(1) # [B, 1, F, T]

        return X_hat


class VRAE(nn.Module):
    def __init__(self):
        super(VRAE, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}\n")

        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)

    def forward(self, x):
        mu_phi, v_phi, z = self.encoder(x)
        y = self.decoder(z)

        return y, z, mu_phi, v_phi

    def fit(self, data_loader, num_epochs=20, lr=1e-3):

        optimizer = optim.Adam(self.parameters(), lr=lr)
        reconstruction_loss_fn = nn.MSELoss(reduction='sum')

        loss_track = []

        self.train()
        print(f"Starting VAE training for {num_epochs} epochs on device: {self.device}")

        for epoch in range(1, num_epochs+1):
            total_loss = 0

            for batch_idx, (mixed, signal, phase) in enumerate(data_loader):

                x = mixed.to(self.device)
                y = signal.to(self.device)

                optimizer.zero_grad()

                y_hat, _, mu, log_var = self.forward(x)

                # loss de reconstruction
                recon_loss = reconstruction_loss_fn(y_hat, y)

                # divergence KL
                kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

                loss = recon_loss + kl_div

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(data_loader.dataset)
            loss_track.append(avg_loss)
            print(f"Epoch {epoch}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

        print("Training complete.")
        return loss_track

    @torch.no_grad()
    def predict(self, test_loader):
        self.eval()

        true_sources = []
        reconstructed_sources = []
        mixed_sources = []
        phase_sources = []
        print("Starting reconstruction on test set")
        for batch_idx, (mixed, signal, phase) in enumerate(test_loader):
            x_mixed = mixed.to(self.device)
            y_target = signal.to(self.device)

            y_hat, _, _, _ = self.forward(x_mixed)

            reconstructed_sources.append(y_hat.cpu())
            true_sources.append(y_target.cpu())
            mixed_sources.append(x_mixed.cpu())
            phase_sources.append(phase.cpu())

        print("Prediction complete.")
        reconstructed_sources = torch.cat(reconstructed_sources, dim=0)
        true_sources = torch.cat(true_sources, dim=0)
        mixed_sources = torch.cat(mixed_sources, dim=0)
        phase_sources = torch.cat(phase_sources, dim=0)
        return reconstructed_sources, true_sources, mixed_sources, phase_sources


class BetaVRAE(VRAE):
    """
    On fait hériter cette classe de la précédente car cette variation de la vAE
    diffère essentiellement par la définition de la fonction de perte
    """
    def __init__(self, beta=2.0, input_dim: int = 257, latent_dim: int = 32):
        super(BetaVRAE, self).__init__()

        # Dans l'article il est suggéré de prendre beta > 1
        self.beta = beta
        print(f"β-VRAE initialized with β={self.beta}")

    def fit(self, data_loader, num_epochs=20, lr=1e-3):
        """
        Override fit() pour inclure β dans la loss, en mode supervisé.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)

        reconstruction_loss_fn = nn.MSELoss(reduction='sum')

        loss_track = []
        recon_track = []
        kl_track = []

        self.train()
        print(f"Starting β-VRAE training (β={self.beta}) for {num_epochs} epochs on device: {self.device}")

        for epoch in range(1, num_epochs+1):
            total_normalized_loss = 0
            total_normalized_recon = 0
            total_normalized_kl = 0

            for batch_idx, (mixed, signal, phase) in enumerate(data_loader):

                x = mixed.to(self.device)
                y = signal.to(self.device)

                optimizer.zero_grad()

                y_hat, _, mu, log_var = self.forward(x)

                B, C, F, T = x.shape

                recon_loss_sum = reconstruction_loss_fn(y_hat, y)
                kl_div_sum = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

                normalization_factor_time = B * T

                recon_loss_normalized = recon_loss_sum / (F * normalization_factor_time)

                # KL Div Normalisée (Moyenne par pas de temps B*T)
                kl_div_normalized = kl_div_sum / normalization_factor_time

                # 3. Perte totale normalisée AVEC β
                loss = recon_loss_normalized + self.beta * kl_div_normalized

                # 4. Rétropropagation
                loss.backward()
                optimizer.step()

                # 5. Accumulation des pertes NORMALISÉES pour l'affichage
                total_normalized_loss += loss.item()
                total_normalized_recon += recon_loss_normalized.item()
                total_normalized_kl += kl_div_normalized.item()

            # Calculer les moyennes de l'époque
            num_batches = len(data_loader)
            avg_loss = total_normalized_loss / num_batches
            avg_recon = total_normalized_recon / num_batches
            avg_kl = total_normalized_kl / num_batches

            # AJOUT : Mise à jour des trackers
            loss_track.append(avg_loss)
            recon_track.append(avg_recon)
            kl_track.append(avg_kl)

            print(f"Epoch {epoch}/{num_epochs}, Avg Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")

        print("Training complete.")
        return {
            'total_loss': loss_track,
            'recon_loss': recon_track,
            'kl_loss': kl_track
        }


class Encoder_ITL_RAE(nn.Module):
    """
    Utilise une couche LSTM bidirectionnelle pour modéliser les dépendances temporelles,
    puis un MLP pour produire le vecteur latent Z.
    """
    def __init__(
        self,
        input_dim: int = 257,  # longueur dimension frequentielle
        latent_dim: int = 32,  # dimension de l'espace latent
    ):
        super(Encoder_ITL_RAE, self).__init__()

        self.dim_z = latent_dim
        hidden_dim = 128        # dimension couche cachée du MLP
        hidden_lstm_dim = 128   # dimension couche cachée du LSTM

        # 1. LSTM Bidirectionnel
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_lstm_dim,
            num_layers=1,
            batch_first=True,  # [B,T,F]
            bidirectional=True
        )

        listm_output_dim = 2 * hidden_lstm_dim  # 2 * 128 = 256

        # 2. MLP final (Sortie unique Z)
        self.mlp = nn.Sequential(
            nn.Linear(listm_output_dim, hidden_dim),
            nn.ReLU(),
            # Sortie = latent_dim (Z)
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        # x est [B,1,F,T]
        B, C, F, T = x.shape

        # 1. Préparation pour LSTM : [B, 1, F, T] -> [B, T, F]
        x_reshape = x.squeeze(1).permute(0, 2, 1)

        lstm_output, _ = self.lstm(x_reshape)  # [B, T, 2*DIM_LSTM]

        z = self.mlp(lstm_output)

        # z est [B, T, Dim_z] (code latent sur la dimension temporelle)
        return z


# Fonctions KDE pour ITL-AE
def gaussian_kernel_pp(x, sigma):
    """
    Kernel G(x_i, x_j ; sigma^2) utilisé pour Vp = qip
    """
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=2)
    return torch.exp(-dist_sq / (2 * sigma**2))


def gaussian_kernel_pq(x, y, sigma):
    """
    Kernel G(x_i, y_j ; 2*sigma^2) utilisé pour Vc = cross_ip
    """
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=2)
    return torch.exp(-dist_sq / (4 * sigma**2))


def qip(x, sigma):
    """
    Quadratic Information Potential Vp
    """
    N = x.size(0)
    K = gaussian_kernel_pp(x, sigma)
    return K.sum() / (N * N)


def cross_ip(x, y, sigma):
    """
    Cross Information Potential Vc
    """
    N_p = x.size(0)
    N_q = y.size(0)
    K = gaussian_kernel_pq(x, y, sigma)
    return K.sum() / (N_p * N_q)


class ITL_RAE(nn.Module):
    """"
    Information-Theoretic Learning Autoencoder
    En plus du changemet de l'encoder du VAE classique, cette variante utilise
    la divergence de Cauchy Schwarz au lieu de la divergence de KL.
    """
    def __init__(self, regul_param, sigma, div_type='CS', input_dim=257, latent_dim=64):
        super(ITL_RAE, self).__init__()
        self.regul_param = regul_param
        self.sigma = sigma
        self.div_type = div_type.upper()  # 'CS' ou 'ED'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}\n")

        self.encoder = Encoder_ITL_RAE(input_dim, latent_dim).to(self.device)
        self.decoder = Decoder(latent_dim, input_dim).to(self.device)
        print(f"ITL_RAE initialized with lambda = {self.regul_param}, sigma = {self.sigma}, Div: {self.div_type}")

    def forward(self, x):
        """
        Forward pass :
        x : [B, 1, F, T]
        """
        # Encode
        z = self.encoder(x)          # [B, T, latent_dim]

        # Decode
        X_hat = self.decoder(z)      # [B, 1, F, T]

        return z, X_hat

    def loss_function(self, Y_true, X_hat, z, prior_samples=None):
        """
        Calcul de la loss ITL-RAE, en choisissant entre CS-Div et ED-Div.
        Y_true : [B,1,F,T] (Signal pur)
        X_hat : [B,1,F,T]
        z : [B,T,latent_dim]
        """

        B, C, F_d, T = X_hat.shape
        normalization_factor = B * C * F_d * T
        recon_loss_sum = F.mse_loss(X_hat, Y_true, reduction='sum')
        recon_loss_normalized = recon_loss_sum / normalization_factor

        if prior_samples is None:
            prior_samples = torch.randn_like(z).to(self.device)

        B, T, D = z.shape
        z_flat = z.reshape(B * T, D)
        p_flat = prior_samples.reshape(B * T, D)

        # AJOUT : Nombre de points latents (B*T)
        num_latent_points = B * T

        # Calcul des trois potentiels (supposé être des sommes sur B*T)
        Vq_sum = qip(z_flat, self.sigma)
        Vp_sum = qip(p_flat, self.sigma)
        Vc_sum = cross_ip(z_flat, p_flat, self.sigma)

        # AJOUT : Normalisation des potentiels par le nombre de points latents
        Vq_mean = Vq_sum / num_latent_points
        Vp_mean = Vp_sum / num_latent_points
        Vc_mean = Vc_sum / num_latent_points

        # 4) Calcul de la Divergence de Régularisation (regul_div)

        if self.div_type == 'ED':
            regul_div_normalized = Vq_mean + Vp_mean - 2 * Vc_mean

        elif self.div_type == 'CS':
            eps = 1e-12
            regul_div_normalized = torch.log((Vq_mean * Vp_mean + eps) / (Vc_mean * Vc_mean + eps))

        else:
            raise ValueError(f"Type de divergence inconnu: {self.div_type}. Doit être 'CS' ou 'ED'.")

        # 5) Total loss
        loss = recon_loss_normalized + self.regul_param * regul_div_normalized
        return loss, recon_loss_normalized, regul_div_normalized

    def fit(self, data_loader, num_epochs=20, lr=1e-3, prior_samples=None):
        """
        Entraîne l'ITL-RAE en utilisant la Divergence de Cauchy-Schwarz ('CS') ou
        la Divergence Euclidienne ('ED') comme régularisation.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Listes pour le suivi des métriques
        loss_track = []
        recon_track = []
        regul_div_track = []

        self.train()

        print(f"Starting ITL-RAE training (Div: {self.div_type}, λ={self.regul_param}, σ={self.sigma}) for {num_epochs} epochs on device: {self.device}")

        for epoch in range(1, num_epochs + 1):
            total_loss = 0
            total_recon = 0
            total_regul_div = 0

            # Utilisation de tqdm pour la barre de progression par époque
            for batch_data in tqdm(data_loader, desc=f"Epoch {epoch}/{num_epochs}"):
                x = batch_data[0].to(self.device)
                y_target = batch_data[1].to(self.device)

                optimizer.zero_grad()

                # Forward pass: z, X_hat = self.forward(x)
                z, x_hat = self.forward(x)

                loss, recon_loss, regul_div = self.loss_function(
                    y_target, x_hat, z, prior_samples=prior_samples
                )

                loss.backward()
                optimizer.step()

                # Accumulation des pertes pour l'époque
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_regul_div += regul_div.item()

            num_batches = len(data_loader)

            avg_loss = total_loss / num_batches
            avg_recon = total_recon / num_batches
            avg_regul_div = total_regul_div / num_batches

            loss_track.append(avg_loss)
            recon_track.append(avg_recon)
            regul_div_track.append(avg_regul_div)

            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Recon: {avg_recon:.4f} | "
                f"Regul ({self.div_type}): {avg_regul_div:.4f}"
            )
        return {
            'total_loss': loss_track,
            'recon_loss': recon_track,
            f'{self.div_type}_loss': regul_div_track,
        }

    @torch.no_grad()
    def predict(self, test_loader, target_index=1):
        """
        Effectue la reconstruction sur le jeu de test.
        Retourne la reconstruction (signal estimé) et la vraie cible (signal pur).
        """
        self.eval()

        true_sources = []
        reconstructed_sources = []
        mixed_sources = []
        phase_sources = []

        print("Starting reconstruction on test set")

        for batch_data in tqdm(test_loader, desc="Reconstruction"):

            x_mixture = batch_data[0].to(self.device)

            # 2. UTILISATION de target_index pour choisir la vraie cible
            y_true = batch_data[target_index].to(self.device)

            # Le tenseur batch_data[2] (S2_Mag) est ignoré si target_index=1
            phase = batch_data[3].to(self.device)

            # Forward pass
            x_hat, _ = self.forward(x_mixture)

            # 3. Stockage des tenseurs sur CPU
            reconstructed_sources.append(x_hat.cpu())
            true_sources.append(y_true.cpu())
            mixed_sources.append(x_mixture.cpu())
            phase_sources.append(phase.cpu())

        print("Prediction complete.")

        # 4. Concaténation et retour
        reconstructed_sources = torch.cat(reconstructed_sources, dim=0)
        true_sources = torch.cat(true_sources, dim=0)
        mixed_sources = torch.cat(mixed_sources, dim=0)
        phase_sources = torch.cat(phase_sources, dim=0)

        self.train()

        return reconstructed_sources, true_sources, mixed_sources, phase_sources
