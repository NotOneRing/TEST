Figure 4.
    1. Demonstration of Effectiveness of DPPO (by Diffusion DPPO in Hopper, Walker, 
    HalfCheetah, Lift, Can, Square, Transport)


    2. Demonstration that all baselines are implemented correctly:
        1. Coverage of all baselines (DRWR, DAWR, DIPO, IDQL, DQL, QSM, RLPD, Cal-QL, IBRL)
        2. If any baseline works well in some dataset, pick the corresponding dataset
            Fulfilling these two criteria, we select: 
                DRWR in Lift
                DAWR in Transport
                DIPO in Walker
                IDQL in Can
                DQL in HalfCheetah
                QSM in Hopper
                RLPD in Kitchen-Complete-v0
                Cal-QL in Kitchen-Partial-v0
                IBRL in Kitchen-Mixed-v0
                




Figure 5. 
    Demonstration of Different Policy Parameterizations:
        1. DPPO-MLP in Square-State, Transport-State and DPPO-ViT-MLP in Square-Pixel, Transport-Pixel
        
        2. Demonstration that all baselines are implemented correctly:
            1. Coverage of all baselines (DPPO-UNet, Gaussian-MLP, Gaussian-Transformer, GMM-MLP, GMM-Transformer) in Square-State, Transport-State. Coverage of all baselines (DPPO-ViT-MLP, Gaussian-ViT-MLP, DPPO-ViT-UNet) in Square-Pixel, Transport-Pixel 
            2. If any baseline works well in some dataset, pick the corresponding dataset
                Fulfilling these two criteria, we select: 
                    Gaussian-MLP in Square-State
                    Gaussian-Transformer in Square-State
                    GMM-MLP in Square-State
                    DPPO-UNet in Transport-State
                    GMM-Transformer in Transport-State

                    Gaussian-ViT-MLP in Square-Pixel
                    DPPO-ViT-UNet in Transport-Pixel
                    DPPO-ViT-UNet in Transport-Pixel




<!-- 
Figure 6. 
    DPPO-UNet 
    Finiture-Bench Tasks: -->









Figure 7.
    Demonstration of the effectiveness of Pretraining
        1. pretraining M1, M2, M3 of DPPO

        2. Demonstration that all baselines are implemented correctly:
            By the similar methodology, we select:
                GMM of M2            
                Gaussian of M3











































