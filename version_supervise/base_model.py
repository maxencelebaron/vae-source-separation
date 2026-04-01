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
                 input_dim: int = 257, # longueur dimension frequentielle
                 latent_dim: int = 32): # dimension de l'espace latent
        super(Encoder, self).__init__()

        self.dim_z = latent_dim
        hidden_dim = 128  # dimension couche cachée du MLP
        hidden_dim2 = 128
        output_dim = 2 * latent_dim

        # MLP:
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x est [B,1,F,T]
        B, C, F, T = x.shape
        x_flat = x.permute(0, 3, 1, 2).reshape(B*T,F) # [B*T,F]

        # MLP
        raw_output = self.mlp(x_flat) # [B*T, 2*DIM_Z]

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
        # Dimension de sortie : N_FREQ_BINS

        # MLP qui ressemble à l'encodeur
        self.mlp = nn.Sequential(
            # Input: [B*T, DIM_Z]
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) # Sortie : [B*T, N_FREQ_BINS]
        )

    def forward(self, z):
        # Entrée: z de dimension : [B, T, Dim_z]
        B, T, D = z.shape

        # Flatten pour le MLP
        z_flat = z.reshape(B * T, D) # [B * T, Dim_z]

        # MLP
        raw_output = self.mlp(z_flat) # [B * T, N_FREQ_BINS]
        X_hat_BTF = raw_output.reshape(B, T, self.output_dim)  # [B, T, F]
        X_hat = X_hat_BTF.permute(0, 2, 1).unsqueeze(1) # [B, 1, F, T]

        return X_hat


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}\n")

        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)

    def forward(self, x):
        mu_phi, v_phi, z =  self.encoder(x)
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


class BetaVAE(VAE):
    """
    On fait hériter cette classe de la précédente car cette variation de la VAE
    diffère essentiellement par la définition de la fonction de perte
    """
    def __init__(self, beta=2.0):
        super(BetaVAE, self).__init__()
        # Dans l'article il est suggérer de prendre beta >1
        self.beta = beta
        print(f"β-VAE initialized with β={self.beta}")

    def fit(self, data_loader, num_epochs=20, lr=1e-3):
        """
        Override fit() pour inclure β dans la loss
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        reconstruction_loss_fn = nn.MSELoss(reduction='sum')

        loss_track = []

        self.train()
        print(f"Starting VAE training for {num_epochs} epochs on device: {self.device}")

        for epoch in range(1, num_epochs+1):
            total_loss = 0

            for batch_data in data_loader:

                x = batch_data[0].to(self.device)

                optimizer.zero_grad()

                x_hat, _, mu, log_var = self.forward(x)

                # loss de reconstruction
                recon_loss = reconstruction_loss_fn(x_hat, x)

                # divergence KL
                kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

                loss = recon_loss + self.beta * kl_div

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(data_loader.dataset)
            loss_track.append(avg_loss)
            print(f"Epoch {epoch}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

        print("Training complete.")
        return loss_track


class Encoder_ITL_AE(nn.Module):
    """
    Encoder pour l'ITL-AE.
    Prend le spectrogramme x et renvoie directement la variable latente z
    """
    def __init__(
        self,
        input_dim: int = 257,   # dimension fréquentielle
        latent_dim: int = 64,   # dimension espace latent
    ):
        super(Encoder_ITL_AE, self).__init__()

        self.dim_z = latent_dim
        hidden_dim = 128

        # L'encodeur ITL-AE doit sortir un vecteur latent de dimension dim_z
        output_dim = latent_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x : [B,1,F,T]
        B, C, F, T = x.shape

        x_flat = x.permute(0, 3, 1, 2).reshape(B*T, F)  # [B*T, F]

        z_flat = self.mlp(x_flat)  # [B*T, DIM_Z]

        z = z_flat.reshape(B, T, self.dim_z)

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


class ITL_AE(nn.Module):
    """"
    Information-Theoretic Learning Autoencoder
    """
    def __init__(self, regul_param, sigma):
        super(ITL_AE, self).__init__()
        self.regul_param = regul_param
        self.sigma = sigma  # bandwidth pour KDE

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}\n")

        self.encoder = Encoder_ITL_AE().to(self.device)
        self.decoder = Decoder().to(self.device)
        print(f"ITL_AE initialized with lambda = {self.regul_param}, sigma = {self.sigma}")

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

    def loss_function(self, X, X_hat, z):
        """
        Calcul de la loss ITL-AE
        X : [B,1,F,T]
        X_hat : [B,1,F,T]
        z : [B,T,latent_dim]
        """

        # 1) Reconstruction loss
        recon_loss = F.mse_loss(X_hat, X)

        # Flatten pour les KDE
        B, T, D = z.shape
        z_flat = z.reshape(B * T, D)
        p_flat = torch.randn_like(z_flat).to(z_flat.device)

        # 3) Divergence
        # Vq = V(z,z) avec noyau σ²
        Vq = qip(z_flat, self.sigma)
        Vp = qip(p_flat, self.sigma)

        # Vc = V(z,p)
        Vc = cross_ip(z_flat, p_flat, self.sigma)

        euc_div = Vq + Vp - 2 * Vc

        # 4) Total loss
        loss = recon_loss + self.regul_param * euc_div

        return loss

    def fit(self, data_loader, num_epochs=20, lr=1e-3):
        """
        Entraîne l'ITL-AE en utilisant la Divergence de Cauchy-Schwarz comme régularisation.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Listes pour le suivi des métriques
        loss_track = []

        self.train()
        print(f"Starting ITL-AE training (λ={self.regul_param}, σ={self.sigma}) for {num_epochs} epochs on device: {self.device}")

        for epoch in range(1, num_epochs + 1):
            total_loss = 0

            for batch_data in data_loader:
                x = batch_data[0].to(self.device)
                optimizer.zero_grad()

                # Forward pass
                # Note: 'forward' de ITL_AE renvoie (z, X_hat)
                z, x_hat = self.forward(x)

                # Calcul de la Loss
                loss = self.loss_function(
                    x, x_hat, z
                )

                loss.backward()
                optimizer.step()

                # Accumulation des pertes pour l'époque
                total_loss += loss.item()

            # Moyennes par epoch
            # La division par le nombre total d'échantillons donne
            # la perte moyenne par échantillon.
            avg_loss = total_loss / len(data_loader.dataset)

            loss_track.append(avg_loss)

            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Loss: {avg_loss:.4f} | "
            )

        print("Training complete.")
        return loss_track

    @torch.no_grad()
    def predict(self, test_loader):
        # Met le modèle en mode évaluation
        self.eval()

        true_sources = []
        reconstructed_sources = []

        print("Starting reconstruction on test set")

        # Utilisation de tqdm pour visualiser la progression
        for batch_data in tqdm(test_loader, desc="Reconstruction"):

            # Le DataLoader retourne un tuple, on prend le premier élément (la magnitude x)
            x_true = batch_data[0].to(self.device)

            # Appel au forward : z contient le code latent, x_hat la reconstruction.
            # Nous n'avons besoin que de x_hat pour la reconstruction.
            z, x_hat = self.forward(x_true)

            # Stockage des résultats sur CPU pour libérer de la mémoire GPU
            reconstructed_sources.append(x_hat.cpu())
            true_sources.append(x_true.cpu())

        print("Prediction complete.")

        # Concaténation de tous les lots pour former les tenseurs finaux
        reconstructed_sources = torch.cat(reconstructed_sources, dim=0)
        true_sources = torch.cat(true_sources, dim=0)

        self.train()

        return reconstructed_sources, true_sources

