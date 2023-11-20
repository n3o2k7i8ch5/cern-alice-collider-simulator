'''
def show_deemb_quality(embedder: PDGEmbedder, deembedder: PDGDeembedder, device):
    prtcl_idxs = torch.tensor(particle_idxs(), device=device)

    pdg_onehot = func.one_hot(
        prtcl_idxs,
        num_classes=PDG_EMB_CNT
    ).float()
    emb = embedder(pdg_onehot)
    one_hot_val = deembedder(emb)
    gen_idxs = torch.argmax(one_hot_val, dim=0)  # .item()  # .unsqueeze(dim=2)

    acc = (torch.eq(prtcl_idxs, gen_idxs) == True).sum(dim=0).item()

    print('Deembeder acc: ' + str(acc) + '/' + str(len(prtcl_idxs)))
'''