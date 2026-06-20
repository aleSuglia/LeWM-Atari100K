import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .modules import SIGReg

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class LeWM(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        reward_head,
        done_head,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.reward_head = reward_head
        self.done_head = done_head
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

        self.sigreg = SIGReg()

    def encode(self, pixels):
        """
        Input: Preprocessed Atari frames
        Output: Encoded frames
        """
        b = pixels.size(0)
        flattened = rearrange(pixels, "b t c h w -> (b t) c h w")                       # (b, t, c, h, w) -> (bt, c, h, w)
        output = self.encoder(flattened, interpolate_pos_encoding=True)                 # (bt, c, h, w) -> (bt, tokens, z)
        emb = output.last_hidden_state[:, 0]                                            # (bt, tokens, d) -> (bt, z)
        emb = self.projector(emb)                                                       # (bt, z) -> (bt, z)
        return rearrange(emb, "(b t) d -> b t d", b=b)                                  # (bt, z) -> (b, t, z)

    def encode_action(self, actions):
        """
        Input: Atari actions (int)
        Output: Encoded action embeddings
        """
        return self.action_encoder(actions)                                             # (b, t) -> (b, t, z)

    def predict(self, emb, act_emb):
        """
        Input: Pixel embeddings, Action embeddings (should match history size)
        Output: Predicted embeddings
        """
        preds = self.predictor(emb, act_emb)                                            # (b, t, z), (b, t, z) -> (b, t, z)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))                    # (b, t, z) -> (bt, z) -> (bt, z)
        return rearrange(preds, "(b t) d -> b t d", b=emb.size(0))                      # (bt, z) -> (b, t, z)

    def predict_reward(self, emb, act_emb):
        """
        Input: Latent, Encoded action
        Output: Predicted scalar reward
        """
        return self.reward_head(emb, act_emb)                                           # (b, t, z), (b, t, z) -> (b, t, 1)

    def predict_done(self, emb, act_emb):
        """
        Input: Latent, Encoded action
        Output: Predicted scalar done flag
        """
        return self.done_head(emb, act_emb)                                             # (b, t, z), (b, t, z) -> (b, t, 1)

    ##################
    ## Interactions ##
    ##################

    def transition(self, emb, act_emb):
        """
        Input: Latent, Encoded action (in accordance with history_size)
        Output: Predicted latent, reward, done flag
        """
        nxt_emb = self.predict(emb, act_emb)[:, -1, :].unsqueeze(1)                     # (b, t, z), (b, t, z) -> (b, z) -> (b, 1, z)
        
        nxt_rew = self.predict_reward(emb, act_emb)[:, -1, :]                           # (b, t, z), (b, t, z) -> (b, 1)
        nxt_don = self.predict_done(emb, act_emb)[:, -1, :]                             # (b, t, z), (b, t, z) -> (b, 1)
        
        return nxt_emb, nxt_rew, nxt_don
    
    def predicted_sequence(self, emb, act_emb, history_size: int = 3):
        """
        Input: List of latents, actions embeddings
        Output: List of predicted latents, rewards, done flags
        """
        _, t, _ = emb.shape
        pred_emb = []
        pred_rew = []
        pred_don = []
        
        for step in range(1, t+1):
            start_idx = max(0, step - history_size)
            
            ctx_emb = emb[:, start_idx:step, :]
            ctx_act = act_emb[:, start_idx:step, :]
            
            nxt_emb, nxt_rew, nxt_don = self.transition(ctx_emb, ctx_act)

            pred_emb.append(nxt_emb)
            pred_rew.append(nxt_rew)
            pred_don.append(nxt_don)

        #### print shapes for verification, need to verify the one-off loss for pred_emb, emb

        return torch.cat(pred_emb, dim=1), torch.cat(pred_rew, dim=1), torch.cat(pred_don, dim=1)

    def loss(
            self,
            observations,
            actions,
            rewards,
            dones,
            loss_weights,
            history_size = 3,
    ):
        """
        Input: Preprocessed observations, actions,
                rewards, dones, loss_weights: dict
        Output: Dictionary containing all loss terms

        need to send rewards and dones as tensors of shape (b, t)
        """
        emb = self.encode(observations)
        act_emb = self.encode_action(actions)
        pred_emb, pred_rew, pred_done = self.predicted_sequence(emb, act_emb, history_size)

        pred_loss = (detach_clone(emb[:, 1:, :]) - pred_emb[:, :-1, :]).pow(2).mean()
        sigreg_loss = self.sigreg(emb.transpose(0, 1))

        rew_loss = F.smooth_l1_loss(pred_rew, detach_clone(rewards))
        don_loss = F.binary_cross_entropy_with_logits(pred_done, detach_clone(dones.float()))

        total_loss = (
            (loss_weights['pred_loss'] * pred_loss) 
            + (loss_weights['sigreg_loss'] * sigreg_loss) 
            + ((loss_weights['rew_loss'] * rew_loss)) 
            + (loss_weights['don_loss'] * don_loss)
        )
        
        losses = {
            'pred_loss': pred_loss,
            'sigreg_loss': sigreg_loss,
            'rew_loss': rew_loss,
            'don_loss': don_loss,
            'total_loss': total_loss
        }

        return losses