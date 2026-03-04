# Enhancing Post-Quantum Cryptographic Protocols with Quantum-Resistant Key Exchange Mechanisms — Implementation in Multilingual Ayurvedic Chatbot

> **Reference Paper:** *"Enhancing Post-Quantum Cryptographic Protocols with Quantum-Resistant Key Exchange Mechanisms"*
> — Gohin Balakrishnan et al., 2025 11th International Conference on Communication and Signal Processing (ICCSP), IEEE
> DOI: 10.1109/ICCSP64183.2025

---

## 1. Title of the Work

**Secure Communication Layer for Multilingual Ayurvedic Chatbot Using Quantum-Resistant Lattice-Based Key Exchange (Q-LREF)**

This work implements the **Quantum-Lattice Resilient Exchange Framework (Q-LREF)** — a post-quantum cryptographic key exchange mechanism based on lattice-based cryptography — to secure the communication channel between the **Frontend (localhost chatbot UI)** and the **Backend Server** of the Multilingual Ayurvedic Health Advisor Chatbot. The Q-LREF protocol utilizes the Learning With Errors (LWE) problem to ensure that all user health queries and Ayurvedic responses are encrypted with quantum-resistant security, protecting sensitive health information against both classical and future quantum adversaries.

---

## 2. Problem Statement

Modern web-based chatbot applications, including health-domain systems like our Multilingual Ayurvedic Chatbot, transmit sensitive user data (health queries, symptoms, personal wellness information) between the **frontend UI** and the **backend server** over HTTP/HTTPS channels. Current encryption standards (RSA, ECC) that secure these channels face a critical and imminent threat:

- **Quantum Computing Threat:** Shor's algorithm can efficiently factor large integers and compute discrete logarithms, breaking RSA-2048 and ECC P-256 — the backbone of today's TLS/HTTPS encryption [2].
- **Grover's Algorithm:** Provides a quadratic speedup for brute-force key search, effectively halving the security of symmetric keys (e.g., AES-128 reduced to 64-bit security) [3].
- **"Harvest Now, Decrypt Later" Attacks:** Adversaries can intercept and store encrypted health data today, then decrypt it once quantum computers become available — particularly dangerous for **medical and health information** which retains sensitivity indefinitely.
- **Healthcare Data Sensitivity:** Ayurvedic health queries may contain personal wellness details (symptoms, dosha types, dietary habits). Inadequate encryption exposes users to privacy violations and potential misuse of health information.

### Why This Matters for Our Chatbot

| Vulnerability | Impact on Ayurvedic Chatbot |
|---|---|
| RSA/ECC broken by Shor's Algorithm | Frontend ↔ Backend TLS channel compromised |
| Brute-force via Grover's Algorithm | Symmetric session keys recoverable faster |
| Harvest-now-decrypt-later | Stored user health data exposed in post-quantum era |
| No end-to-end encryption at app layer | Server-level TLS doesn't protect against server-side breaches |

**The problem:** There is no application-layer quantum-resistant encryption between the chatbot frontend and backend, leaving user health data vulnerable to both current sophisticated attacks and future quantum attacks.

**The solution:** Implement the Q-LREF lattice-based key exchange protocol at the **application layer** between the frontend chatbot UI and the backend server to establish quantum-resistant shared secrets for encrypting all health-related communications.

---

## 3. Architecture Diagram

### System Architecture: Q-LREF Secured Ayurvedic Chatbot

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Localhost Chatbot UI)                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐     │
│  │  User Interface (Streamlit / Flask)                                         │     │
│  │  ┌───────────────┐    ┌──────────────────────────────────────┐             │     │
│  │  │  User Input    │───▶│  Q-LREF Client Module                │             │     │
│  │  │  (Hindi/EN)    │    │  ┌────────────────────────────────┐  │             │     │
│  │  └───────────────┘    │  │ 1. KeyGen(λ, n, q, χ)         │  │             │     │
│  │                        │  │    - Private key: s_A ← χⁿ    │  │             │     │
│  │                        │  │    - Matrix A ← Z_q^{n×n}     │  │             │     │
│  │                        │  │    - Error: e_A ← χⁿ          │  │             │     │
│  │                        │  │    - Public key: p_A = As+e    │  │             │     │
│  │                        │  │ 2. Shared Secret Derivation    │  │             │     │
│  │                        │  │    K_A = S_A · P_B + e (mod q) │  │             │     │
│  │                        │  │ 3. AES Encryption with K       │  │             │     │
│  │                        │  └────────────────────────────────┘  │             │     │
│  │                        └──────────────┬───────────────────────┘             │     │
│  └───────────────────────────────────────┼─────────────────────────────────────┘     │
└──────────────────────────────────────────┼──────────────────────────────────────────┘
                                           │
                         ┌─────────────────▼─────────────────┐
                         │   SECURE CHANNEL (Q-LREF)          │
                         │   ┌─────────────────────────────┐  │
                         │   │ Exchange Public Keys (p_A,   │  │
                         │   │ p_B) over network            │  │
                         │   │ ─────────────────────────    │  │
                         │   │ Encrypted Payload:           │  │
                         │   │ AES-256(shared_secret, data) │  │
                         │   └─────────────────────────────┘  │
                         └─────────────────┬─────────────────┘
                                           │
┌──────────────────────────────────────────┼──────────────────────────────────────────┐
│                        BACKEND SERVER                                                │
│  ┌───────────────────────────────────────┼─────────────────────────────────────┐     │
│  │                        ┌──────────────▼──────────────────────┐              │     │
│  │                        │  Q-LREF Server Module                │              │     │
│  │                        │  ┌────────────────────────────────┐  │              │     │
│  │                        │  │ 1. KeyGen(λ, n, q, χ)         │  │              │     │
│  │                        │  │    - Generate (p_B, s_B)       │  │              │     │
│  │                        │  │ 2. Shared Secret Derivation    │  │              │     │
│  │                        │  │    K_B = S_B · P_A + e (mod q) │  │              │     │
│  │                        │  │ 3. Reconcile + Verify (Hash)   │  │              │     │
│  │                        │  │ 4. AES Decryption with K       │  │              │     │
│  │                        │  └────────────────────────────────┘  │              │     │
│  │                        └──────────────┬──────────────────────┘              │     │
│  │                                       ▼                                     │     │
│  │  ┌────────────────────────────────────────────────────────────┐             │     │
│  │  │            NLU + RAG Pipeline (Existing)                    │             │     │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐ │             │     │
│  │  │  │ Language  │─▶│ Hindi    │─▶│ FAISS    │─▶│ mT5-small │ │             │     │
│  │  │  │ Detection │  │ Preproc  │  │ Retrieval│  │ Generator │ │             │     │
│  │  │  └──────────┘  └──────────┘  └──────────┘  └───────────┘ │             │     │
│  │  └────────────────────────────────────────────────────────────┘             │     │
│  └─────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐     │
│  │  Key Management Service (KMS)                                               │     │
│  │  • Secure key generation (quantum-resistant)                                │     │
│  │  • Key distribution & rotation                                              │     │
│  │  • Key revocation for compromised keys                                      │     │
│  └─────────────────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### Q-LREF Key Exchange Flow (Frontend ↔ Backend)

```
    Frontend (Client)                                   Backend (Server)
    ─────────────────                                   ─────────────────
          │                                                    │
          │  1. KeyGen(λ=128, n=512, q, χ)                     │
          │     s_A ← χⁿ (private key)                         │
          │     A ← Z_q^{n×n}                                  │
          │     e_A ← χⁿ (error vector)                        │
          │     p_A = A·s_A + e_A (mod q)                       │
          │                                                    │
          │  ─────── Send Public Key p_A ──────────▶           │
          │                                                    │
          │                      2. KeyGen(λ=128, n=512, q, χ) │
          │                         s_B ← χⁿ (private key)     │
          │                         e_B ← χⁿ                   │
          │                         p_B = A·s_B + e_B (mod q)   │
          │                                                    │
          │           ◀────── Send Public Key p_B ─────────     │
          │                                                    │
          │  3. Shared Secret:                                  │
          │     e'_A ← χⁿ                                      │
          │     K_A = s_A · p_B + e'_A (mod q)                  │
          │                                                    │
          │                         4. Shared Secret:           │
          │                            e'_B ← χⁿ               │
          │                            K_B = s_B · p_A + e'_B   │
          │                                                    │
          │  5. Hash verification:                              │
          │     h_A = Hash(K_A)                                 │
          │  ─────── Exchange Hashes ──────────────▶            │
          │           ◀──────────────────────────────           │
          │                                                    │
          │  6. If h_A == h_B:                                  │
          │     K = Reconcile(K_A, K_B)                         │
          │     ✅ Shared secret established!                    │
          │                                                    │
          │  ═══════ AES-256-GCM(K, user_query) ══════▶        │
          │           ◀══════ AES-256-GCM(K, response) ════════ │
          │                                                    │
```

---

## 4. Solution Methodology

### Overview

We integrate the **Q-LREF (Quantum-Lattice Resilient Exchange Framework)** into the Ayurvedic Chatbot as an **application-layer encryption module** that sits between the frontend chatbot UI and the backend NLU/RAG server. The methodology follows the research paper's lattice-based key exchange protocol, adapted for a web client-server architecture.

### Core Principles

| Principle | Implementation |
|-----------|---------------|
| **Security** | LWE-based lattice cryptography — conjectured quantum-resistant |
| **Efficiency** | Optimized key generation and shared secret derivation for low-latency web communication |
| **Scalability** | Modular KMS design allowing key rotation and multi-session handling |

### Phase-by-Phase Implementation

#### Phase 1: Key Generation (Both Client & Server)

Each party independently generates a key pair using the LWE-based `KeyGen` algorithm:

```
Algorithm KeyGen(λ, n, q, χ):
    Generate random private key s ← χⁿ
    Generate random matrix A ← Z_q^{n×n}
    Generate error vector e ← χⁿ
    Compute public key p = A·s + e (mod q)
    Return (p, s)
```

**Parameters (aligned with NIST Security Level 3):**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Lattice dimension (n) | 512 | 192-bit classical security |
| Noise distribution (σ) | 3.2 | Gaussian — balances hardness and correctness |
| Modulus (q) | Large prime | Defines the lattice ring |
| Security parameter (λ) | 128 | Minimum security level |

#### Phase 2: Public Key Exchange

- Frontend sends `p_A` to the backend server via an initial handshake HTTP request.
- Backend responds with `p_B`.
- Both parties now hold each other's public keys.

#### Phase 3: Shared Secret Derivation

Each party computes a shared secret using their private key and the other party's public key:

```
Frontend:  K_A = S_A · P_B + e_A (mod q)
Backend:   K_B = S_B · P_A + e_B (mod q)
```

Due to the mathematical properties of LWE: `K_A ≈ K_B` (approximately equal, differing only by small noise terms).

#### Phase 4: Reconciliation & Verification

```
Algorithm Reconcile(K_A, K_B):
    Define reconciliation boundaries b
    For each coefficient i in K_A, K_B:
        diff = |K_A[i] - K_B[i]| (mod q)
        If diff < b:
            K[i] = Round(K_A[i])
        Else:
            Return Error("Reconciliation failed")
    Return K
```

Both parties hash their derived secrets and exchange hashes to verify correctness. Upon successful verification, the reconciled shared secret `K` is used as the symmetric key for **AES-256-GCM** encryption of all subsequent messages.

#### Phase 5: Encrypted Communication

```
Frontend → Backend:  AES-256-GCM(K, user_health_query)
Backend → Frontend:  AES-256-GCM(K, ayurvedic_response)
```

All user queries (Hindi/English health questions) and chatbot responses (Ayurvedic advice) are encrypted end-to-end at the application layer.

### Security Reduction

The security formally reduces to the **hardness of the LWE problem**: if an adversary can break the key exchange, they can solve LWE — which is widely conjectured to be quantum-resistant. This provides provable security guarantees beyond what RSA/ECC offer against quantum computers.

### Integration with Existing Chatbot Pipeline

```
User Input → [Q-LREF Encrypt] → Network → [Q-LREF Decrypt] → Language Detection
   → Hindi Preprocessing → FAISS Retrieval → mT5 Generation → [Q-LREF Encrypt]
   → Network → [Q-LREF Decrypt] → Display Response
```

The Q-LREF module is **transparent** to the NLU/RAG pipeline — it operates as a secure wrapper around the existing HTTP communication, requiring no changes to the core Ayurvedic chatbot logic.

---

## 5. Metrics Chosen to Evaluate the Performance

### A. Security Metrics

| Metric | Method | Target |
|--------|--------|--------|
| **Resistance to Shor's Algorithm** | LWE hardness test — simulate quantum solvers against lattice dimensions (256, 512, 1024) | 0% key recovery success |
| **Resistance to Grover's Algorithm** | Brute-force search simulation for key sizes (128, 192, 256-bit) | Attack time ≥ 10⁹ years |
| **MITM Attack Resistance** | Simulated man-in-the-middle interception and shared secret reconstruction | 0% success rate |
| **Brute-Force Key Recovery** | Classical and quantum brute-force simulations with key space 2²⁵⁶ | Infeasible (≥10¹² years classical, ≥10⁶ years quantum) |

**Expected Results (from paper):**

| Lattice Dimension (n) | Key Size (bits) | Quantum Attack Success | Computation Time |
|---|---|---|---|
| 256 | 128 | 0% | 10¹² s (Infeasible) |
| 512 | 256 | 0% | 10¹⁸ s (Infeasible) |
| 1024 | 512 | 0% | 10²⁴ s (Infeasible) |

### B. Performance Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Key Generation Time** | Time to generate LWE key pair on client/server | ≤ 50 ms |
| **Key Exchange Latency** | Total time for the Q-LREF handshake (keygen + exchange + verify) | ≤ 200 ms |
| **Encryption/Decryption Overhead** | Additional time per message for AES-256-GCM with Q-LREF-derived key | ≤ 1 ms per message |
| **Key Size (bytes)** | Size of public key transmitted over network | ~1024 bytes (comparable to CRYSTALS-Kyber's 1184 bytes) |
| **Total End-to-End Latency** | User query to response time (including crypto overhead) | ≤ 3.5 seconds (vs ≤ 3s without crypto) |
| **Memory Overhead** | Additional RAM usage for crypto operations on client/server | ≤ 10 MB |

**Comparison with Classical Approaches (from paper):**

| Metric | RSA-2048 | ECC P-256 | CRYSTALS-Kyber | Q-LREF (Ours) |
|--------|----------|-----------|----------------|---------------|
| Key Size (bytes) | 256 | 32 | 1184 | 1024 |
| Encryption (ms) | 0.98 | 0.31 | 0.12 | 0.18 |
| Decryption (ms) | 16.25 | 0.45 | 0.15 | 0.22 |
| Security Level | Quantum Vulnerable | Quantum Vulnerable | Quantum-Resistant | Quantum-Resistant |
| NIST Level | N/A | N/A | Level 3 | Level 3 |

### C. Threat Model Evaluation

| Attack Type | Classical Feasibility | Quantum Feasibility | Status |
|---|---|---|---|
| Man-in-the-Middle (MITM) | High Risk | High Risk | **Mitigated** |
| Brute-Force Attack | Very High Risk | Moderate Risk | **Mitigated** |
| Quantum Key Recovery | Not Feasible | Not Feasible | **Mitigated** |

### D. Chatbot-Specific Joint Metrics

| Metric | Purpose | Target |
|--------|---------|--------|
| **Crypto-Inclusive Response Time** | End-to-end latency with Q-LREF encryption active | ≤ 3.5 seconds |
| **Session Establishment Success Rate** | Percentage of successful Q-LREF handshakes | ≥ 99.9% |
| **Key Rotation Frequency** | How often session keys are refreshed | Every 30 minutes or 100 messages |
| **Data Integrity Verification** | Percentage of messages passing hash verification | 100% |

---

## References

[1] D. J. Bernstein, J. Buchmann, and E. Dahmen, *Post-Quantum Cryptography*. Springer, 2009.
[2] P. W. Shor, "Algorithms for quantum computation," FOCS, 1994, pp. 124–134.
[3] L. K. Grover, "A fast quantum mechanical algorithm for database search," STOC, 1996, pp. 212–219.
[4] NIST, "Post-quantum cryptography standardization," 2022.
[5] C. Peikert, "Lattice cryptography for the internet," PQCrypto, 2014, pp. 197–219.
[6] J. Hoffstein et al., "NTRU: A ring-based public key cryptosystem," ANTS, 1998, pp. 267–288.
[7] D. Jao and L. De Feo, "Towards quantum-resistant cryptosystems from supersingular elliptic curve isogenies," PQCrypto, 2011, pp. 19–34.
[8] V. Vaikuntanathan, "A tutorial on learning with errors," FOCS, 2017, pp. 558–578.
[9] W. Diffie and M. E. Hellman, "New directions in cryptography," IEEE Trans. Inf. Theory, 1976.
