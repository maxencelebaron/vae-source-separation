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
    def __init__(self, beta=2.0, input_dim: int = 257, latent_dim: int = 32):
        super(BetaVRAE, self).__init__()

        # Dans l'article il est suggéré de prendre beta > 1
        self.beta = beta
        print(f"β-VRAE initialized with β={self.beta}")

    def fit(self, data_loader, num_epochs=20, lr=1e-3):
        """
        Override fit() pour inclure β dans la loss.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)

        reconstruction_loss_fn = nn.MSELoss(reduction='sum')

        loss_track = []

        self.train()
        print(f"Starting β-VRAE training (β={self.beta}) for {num_epochs} epochs on device: {self.device}")

        for epoch in range(1, num_epochs+1):
            total_normalized_loss = 0

            for batch_idx, (mixed, signal, phase) in enumerate(data_loader):

                x = mixed.to(self.device)
                y = signal.to(self.device)

                optimizer.zero_grad()

                y_hat, _, mu, log_var = self.forward(x)

                B, C, F, T = x.shape

                recon_loss = reconstruction_loss_fn(y_hat, y)
                kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

                # 3. Perte totale normalisée AVEC β
                loss = recon_loss + self.beta * kl_div

                # 4. Rétropropagation
                loss.backward()
                optimizer.step()

                # 5. Accumulation des pertes NORMALISÉES pour l'affichage
                total_normalized_loss += loss.item()

            # Calculer les moyennes de l'époque
            avg_loss = total_normalized_loss / len(data_loader.dataset)

            # AJOUT : Mise à jour des trackers
            loss_track.append(avg_loss)

            print(f"Epoch {epoch}/{num_epochs}, Avg Loss: {avg_loss:.4f} ")

        print("Training complete.")
        return loss_track


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
    """
    def __init__(self, regul_param, sigma, input_dim=257, latent_dim=32):
        super(ITL_RAE, self).__init__()
        self.regul_param = regul_param
        self.sigma = sigma

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}\n")

        self.encoder = Encoder_ITL_RAE(input_dim, latent_dim).to(self.device)
        self.decoder = Decoder(latent_dim, input_dim).to(self.device)
        print(f"ITL_RAE initialized with lambda = {self.regul_param}, sigma = {self.sigma}")

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

    def loss_function(self, Y_true, X_hat, z):
        """
        Calcul de la loss ITL-RAE.
        Y_true : [B,1,F,T] (Signal pur)
        X_hat : [B,1,F,T]
        z : [B,T,latent_dim]
        """

        B, C, F_d, T = X_hat.shape
        recon_loss = F.mse_loss(X_hat, Y_true, reduction='sum')

        prior_samples = torch.randn_like(z).to(self.device)

        B, T, D = z.shape
        z_flat = z.reshape(B * T, D)
        p_flat = prior_samples.reshape(B * T, D)

        # Calcul des trois potentiels (supposé être des sommes sur B*T)
        Vq = qip(z_flat, self.sigma)
        Vp = qip(p_flat, self.sigma)
        Vc = cross_ip(z_flat, p_flat, self.sigma)

        # 4) Calcul de la Divergence de Régularisation
        regul_div_normalized = Vq + Vp - 2 * Vc

        # 5) Total loss
        loss = recon_loss + self.regul_param * regul_div_normalized
        return loss

    def fit(self, data_loader, num_epochs=20, lr=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Listes pour le suivi des métriques
        loss_track = []

        self.train()

        print(f"Starting ITL-RAE training (λ={self.regul_param}, σ={self.sigma}) for {num_epochs} epochs on device: {self.device}")

        for epoch in range(1, num_epochs + 1):
            total_loss = 0

            # Utilisation de tqdm pour la barre de progression par époque
            for batch_data in tqdm(data_loader, desc=f"Epoch {epoch}/{num_epochs}"):
                x = batch_data[0].to(self.device)
                y_target = batch_data[1].to(self.device)

                optimizer.zero_grad()

                # Forward pass: z, X_hat = self.forward(x)
                z, x_hat = self.forward(x)

                loss = self.loss_function(
                    y_target, x_hat, z
                )

                loss.backward()
                optimizer.step()

                # Accumulation des pertes pour l'époque
                total_loss += loss.item()

            num_batches = len(data_loader)

            avg_loss = total_loss / num_batches

            loss_track.append(avg_loss)
            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Loss: {avg_loss:.4f} | "
            )
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
            x_mixture = mixed.to(self.device)

            # y_true est le signal pur (target)
            y_true = signal.to(self.device)

            # Phase associée
            phase = phase.to(self.device)

            # Forward pass
            _, x_hat = self.forward(x_mixture)

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
