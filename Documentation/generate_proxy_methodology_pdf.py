from pathlib import Path
from html import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "proxy_pricing_methodology.pdf"


def page_footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D5D9DE"))
    canvas.line(0.65 * inch, 0.52 * inch, 7.85 * inch, 0.52 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#606872"))
    canvas.drawString(0.68 * inch, 0.34 * inch, "Monte Carlo Proxy Pricing Methodology")
    canvas.drawRightString(7.82 * inch, 0.34 * inch, f"Page {document.page}")
    canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=25,
        leading=30,
        textColor=colors.HexColor("#17324D"),
        alignment=TA_CENTER,
        spaceAfter=18,
    )
)
styles.add(
    ParagraphStyle(
        name="Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=17,
        textColor=colors.HexColor("#52606D"),
        alignment=TA_CENTER,
        spaceAfter=14,
    )
)
styles.add(
    ParagraphStyle(
        name="H1Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#17324D"),
        spaceBefore=4,
        spaceAfter=10,
    )
)
styles.add(
    ParagraphStyle(
        name="H2Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#24557A"),
        spaceBefore=10,
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=13.2,
        textColor=colors.HexColor("#20262D"),
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="SmallCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#4A5560"),
        spaceAfter=4,
    )
)
styles.add(
    ParagraphStyle(
        name="CodeCustom",
        fontName="Courier",
        fontSize=7.7,
        leading=10.4,
        leftIndent=8,
        rightIndent=8,
        borderColor=colors.HexColor("#D9E2EA"),
        borderWidth=0.6,
        borderPadding=7,
        backColor=colors.HexColor("#F6F8FA"),
        spaceBefore=4,
        spaceAfter=8,
    )
)
styles.add(
    ParagraphStyle(
        name="MathCustom",
        parent=styles["BodyText"],
        fontName="Times-Italic",
        fontSize=9.2,
        leading=13.4,
        leftIndent=2,
        rightIndent=2,
        textColor=colors.HexColor("#17212B"),
        spaceAfter=0,
    )
)
styles.add(
    ParagraphStyle(
        name="Callout",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9.2,
        leading=13,
        textColor=colors.HexColor("#17324D"),
        borderColor=colors.HexColor("#5AA6A6"),
        borderWidth=1,
        borderPadding=8,
        backColor=colors.HexColor("#EDF7F6"),
        spaceBefore=6,
        spaceAfter=10,
    )
)
styles.add(
    ParagraphStyle(
        name="TheoremCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.0,
        leading=12.8,
        textColor=colors.HexColor("#20262D"),
        borderColor=colors.HexColor("#9BB7CC"),
        borderWidth=0.7,
        borderPadding=7,
        backColor=colors.HexColor("#F3F7FA"),
        spaceBefore=5,
        spaceAfter=6,
    )
)


def p(text, style="BodyCustom"):
    return Paragraph(text, styles[style])


def h1(text):
    return Paragraph(text, styles["H1Custom"])


def h2(text):
    return Paragraph(text, styles["H2Custom"])


def eq(text):
    lines = [line.rstrip() for line in text.strip().splitlines()]
    body = "<br/>".join(
        escape(line).replace(" ", "&nbsp;") if line else "&nbsp;"
        for line in lines
    )
    item = Table(
        [[Paragraph(body, styles["MathCustom"])]],
        colWidths=[6.75 * inch],
        hAlign="LEFT",
    )
    item.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FAFBFC")),
                ("BOX", (0, 0), (-1, -1), 0.45, colors.HexColor("#B8C5D1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return item


def theorem(title, text):
    return p(f"<b>Theorem ({title}).</b> {text}", "TheoremCustom")


def proof(text):
    return p(f"<b>Proof.</b> {text}")


def table(data, widths=None, header=True, font_size=7.6):
    item = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDEAF3")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17324D")),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2.5),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7D0D9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row in range(1, len(data)):
        if row % 2 == 0:
            commands.append(
                ("BACKGROUND", (0, row), (-1, row), colors.HexColor("#F7F9FB"))
            )
    item.setStyle(TableStyle(commands))
    return item


def bullet(text):
    return p(f"- {text}")


def add_plot(story, relative_path, caption, max_width=7.0 * inch, max_height=4.8 * inch):
    path = ROOT / relative_path
    if not path.exists():
        return
    image = Image(str(path))
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    story.append(
        KeepTogether(
            [
                Spacer(1, 6),
                image,
                p(caption, "SmallCustom"),
            ]
        )
    )


story = []

# Title
story.extend(
    [
        Spacer(1, 0.75 * inch),
        Paragraph("Monte Carlo Proxy Pricing", styles["TitleCustom"]),
        Paragraph(
            "A step-by-step methodology for European, American, Asian, barrier, cliquet, "
            "ten-asset basket Asian, SLV cliquet, and basket cliquet instruments",
            styles["Subtitle"],
        ),
        Paragraph(
            "Written for a second-year undergraduate with calculus, probability, and linear algebra",
            styles["Subtitle"],
        ),
        Spacer(1, 0.25 * inch),
        p(
            "This guide derives the transformations, estimators, regression bases, "
            "interpolants, variance-reduction methods, dynamic-programming equations, "
            "and validation metrics used in the proxy-pricing experiments.",
            "Callout",
        ),
        Spacer(1, 0.25 * inch),
        table(
            [
                ["Instrument", "Effective state", "Current preferred fitter"],
                ["European call", "1D d1-like moneyness", "PCHIP default; Bernstein optimized"],
                ["American put", "Spot and exercise time", "PCHIP continuation recursion"],
                ["Arithmetic Asian", "1D adjusted moneyness after reduction", "PCHIP default; Akima optimized"],
                [
                    "10-asset basket Asian",
                    "10 spots plus running basket sum",
                    "Moment baseline + PCA sparse-Chebyshev correction",
                ],
                ["Barrier call", "Spot plus alive/hit flag", "PCHIP + Brownian bridge"],
                ["GBM cliquet", "1D accrued clipped return", "Bounded Chebyshev degree 19"],
                ["Single-name SLV cliquet", "Accrued, spot, variance", "Local/spectral hybrid"],
                ["3-asset SLV basket cliquet", "Accrued, 3 spots, 3 variances", "Grouped labels + PCA; order-statistic enrichment needed"],
            ],
            widths=[1.45 * inch, 2.55 * inch, 2.85 * inch],
        ),
        Spacer(1, 0.25 * inch),
        p("Research implementation: proxy_pricing repository", "Subtitle"),
        PageBreak(),
    ]
)

# Contents and scope
story.extend(
    [
        h1("1. Scope and guiding principle"),
        p(
            "A pricing proxy is a fast approximation V_hat(x,t) to an expensive "
            "risk-neutral value V(x,t). The central difficulty is not regression alone. "
            "It is choosing a Markov state, transforming the payoff geometry, generating "
            "unbiased low-noise labels, fitting without violating known bounds, and "
            "testing in the tails."
        ),
        p(
            "The experiments support one broad rule: use product structure to reduce "
            "dimension first, then choose the simplest smoother compatible with the "
            "remaining geometry. A one-dimensional curve and a seven-dimensional SLV "
            "surface should not be forced into the same numerical representation."
        ),
        h2("How to read this guide"),
        p(
            "The main chapters explain the ideas operationally: what is being priced, what "
            "state is needed, how Monte Carlo labels are generated, and how the proxy is "
            "fitted. Proofs are postponed to the appendices so that the pricing workflow "
            "remains visible. Every theorem used by the implementation is restated there "
            "with assumptions, a proof or proof sketch, and its role in the code."
        ),
        p(
            "Appendix H is a mathematical audit of the implementation tricks. It states "
            "the exact claim behind state enrichment, Sobol batching, PCA compression, "
            "moment-matched basket baselines, ridge residual fitting, log-factor targets, "
            "PCHIP residual calibration, relative-error floors, and exact payoff wings."
        ),
        h2("Minimal prerequisites and notation"),
        table(
            [
                ["Symbol", "Meaning"],
                ["S_t", "Underlying asset price at time t"],
                ["r, q, sigma", "Interest rate, dividend yield, volatility"],
                ["E[. | X_t=x]", "Conditional expected value given today's state"],
                ["V(t,x)", "True model price; V_hat(t,x) is its proxy"],
                ["N", "Number of Monte Carlo paths"],
                ["SE", "Estimated Monte Carlo standard error"],
                ["clip(u,L,U)", "min(max(u,L),U)"],
            ],
            widths=[1.6 * inch, 5.2 * inch],
        ),
        h2("Contents"),
        table(
            [
                ["Part", "Topic"],
                ["I", "Risk-neutral valuation, Monte Carlo labels, and variance reduction"],
                ["II", "Target transforms, Chebyshev/Bernstein bases, PCHIP, Akima, Bezier"],
                ["III", "European and Asian one-dimensional reductions"],
                ["IV", "Ten-asset basket Asian, American stopping, and barrier options"],
                ["V", "Single-name cliquet and SLV extensions"],
                ["VI", "Basket Asian and basket cliquet high-dimensional lessons"],
                ["VII", "Generic high-dimensional workflow and validation checklist"],
                ["Appendices", "Theorem statements, proofs, and references"],
            ],
            widths=[0.7 * inch, 6.1 * inch],
        ),
        h2("What the reported error means"),
        p(
            "Unless stated otherwise, relative error is abs(proxy-benchmark) divided by "
            "max(abs(benchmark), 0.01). The 0.01 floor prevents an economically tiny "
            "absolute discrepancy near zero from dominating a percentage statistic. "
            "Signed and absolute errors are retained separately."
        ),
        PageBreak(),
    ]
)

# Foundation
story.extend(
    [
        h1("2. Risk-neutral valuation foundation"),
        h2("2.1 Conditional expectation"),
        p(
            "Under a risk-neutral measure Q, with short rate r and state X_t, a European-"
            "style discounted payoff H has value"
        ),
        eq(
            """
V(t,x) = E_Q[ exp(-Integral_t^T r_u du) H(X_T) | X_t = x ].
"""
        ),
        p(
            "Path-dependent contracts become Markov after augmenting X_t with sufficient "
            "path statistics: a running sum for an Asian, accrued clipped return for a "
            "cliquet, and current variances for SLV."
        ),
        h2("2.2 GBM transition"),
        p("For constant parameters under Q:"),
        eq(
            """
dS_t / S_t = (r-q) dt + sigma dW_t.

Integrating d log(S_t) gives
log(S_T/S_t) = (r-q-sigma^2/2) tau + sigma sqrt(tau) Z,
Z ~ N(0,1).

Therefore
S_T = S_t exp((r-q-sigma^2/2) tau + sigma sqrt(tau) Z).
"""
        ),
        p(
            "This exact transition removes time-discretization error for GBM European, "
            "Asian-fixing, and GBM cliquet simulations."
        ),
        h2("2.3 SLV transition"),
        eq(
            """
dS_i/S_i = (r-q_i) dt + L_i(S_i) sqrt(v_i) dW_i^S
dv_i     = kappa_i(theta_i-v_i) dt + xi_i sqrt(v_i) dW_i^v.
"""
        ),
        p(
            "The experiments use full-truncation Euler. With v_plus=max(v,0):"
        ),
        eq(
            """
S_next = S exp((r-q-0.5 L(S)^2 v_plus) dt
               + L(S) sqrt(v_plus dt) Z_S)

v_next = max(v + kappa(theta-v_plus) dt
               + xi sqrt(v_plus dt) Z_v, 0).
"""
        ),
        p(
            "This preserves positivity numerically but introduces discretization bias. "
            "Proxy error is measured against a benchmark using the same grid, so Euler "
            "bias is outside the reported proxy metric."
        ),
        PageBreak(),
        h2("2.4 Universal pricing-label workflow"),
        p(
            "For every instrument, the expensive training label is constructed before any "
            "proxy is fitted. At state x_i and valuation date t, the reusable sequence is:"
        ),
        table(
            [
                ["Step", "Pricing action", "Output"],
                ["1", "Reconstruct the sufficient Markov state", "x_i"],
                ["2", "Simulate risk-neutral transitions or paths", "X^(j)"],
                ["3", "Evaluate path payoff and discount", "Y_i^(j)"],
                ["4", "Apply unbiased controls or likelihood ratios", "Y_tilde_i^(j)"],
                ["5", "Average paths and estimate sampling error", "V_i and SE_i"],
                ["6", "Repeat on independent validation states", "Benchmark surface"],
            ],
            widths=[0.55 * inch, 4.1 * inch, 2.1 * inch],
        ),
        eq(
            """
Training label:
V_i = (1/N_i) Sum_(j=1)^N_i Y_tilde_i^(j)

Estimated label variance:
Var(V_i) = sample_var(Y_tilde_i) / N_i.
"""
        ),
        p(
            "A proxy approximates the conditional expectation represented by these labels; "
            "it does not replace the risk-neutral pricing definition. Exact payoff regions "
            "are evaluated analytically and removed from the regression whenever possible."
        ),
        PageBreak(),
    ]
)

# MC and variance reduction
story.extend(
    [
        h1("3. Monte Carlo labels and variance reduction"),
        h2("3.1 Plain estimator and uncertainty"),
        eq(
            """
Y_j = discounted payoff on path j
V_hat_MC = (1/N) Sum_j Y_j
SE(V_hat_MC) = sample_std(Y) / sqrt(N).
"""
        ),
        p(
            "The standard error falls only as N^(-1/2). Ten times smaller noise requires "
            "one hundred times as many paths, so structural variance reduction is more "
            "valuable than path count alone."
        ),
        h2("3.2 Antithetic sampling"),
        eq(
            """
For Z ~ N(0,I), both Z and -Z have the same law.
Use pair average: Y_pair = (Y(Z)+Y(-Z))/2.

Var(Y_pair) = 0.5 Var(Y) + 0.5 Cov(Y(Z),Y(-Z)).
"""
        ),
        p(
            "For monotone payoffs the covariance is often negative, reducing variance "
            "without changing the expectation."
        ),
        h2("3.3 Likelihood-ratio importance sampling"),
        p(
            "Suppose the standard normal is sampled from q=N(mu,I) instead of "
            "p=N(0,I). The density ratio follows directly by completing the square:"
        ),
        eq(
            """
p(z) / q(z)
= exp(-z'z/2 + (z-mu)'(z-mu)/2)
= exp(-mu'z + mu'mu/2).

E_p[H(Z)] = E_q[ H(Z) p(Z)/q(Z) ].
"""
        ),
        p(
            "For multiple independent steps, log likelihoods add. The European and "
            "standalone one-dimensional Asian defaults use shifted terminal or path "
            "normals where appropriate. SLV cliquets use a defensive mixture: half "
            "unshifted and half shifted paths."
        ),
        eq(
            """
q_mix(z) = 0.5 p(z) + 0.5 p(z-mu)

p(z)/q_mix(z)
= 1 / [0.5 + 0.5 exp(mu'z - mu'mu/2)].
"""
        ),
        p(
            "The unshifted component protects central states; the shifted component "
            "stabilizes rare positive payoffs near a global floor."
        ),
        p(
            "The ten-asset basket Asian experiment uses both ideas. Its state design is "
            "enriched across basket level, running sum, and PCA directions. Its path "
            "generator also uses a true two-component Gaussian likelihood-ratio mixture "
            "for the long-dated OTM slice where the shift reduced variance. Later fixing "
            "dates use plain Sobol because the same shift increased weighted variance."
        ),
        PageBreak(),
    ]
)

story.extend(
    [
        h1("4. Control variates"),
        p(
            "Let Y be the target payoff and C a correlated control with known expectation "
            "c. Define Y_beta=Y-beta(C-c). It remains unbiased because E[C-c]=0."
        ),
        eq(
            """
Var(Y_beta) = Var(Y) + beta^2 Var(C) - 2 beta Cov(Y,C).

Differentiate with respect to beta:
2 beta Var(C) - 2 Cov(Y,C) = 0.

beta_star = Cov(Y,C) / Var(C).
"""
        ),
        p(
            "The Asian benchmark uses the exact discrete geometric Asian value as the "
            "control expectation. The GBM cliquet uses the future clipped-return sum, "
            "whose first moment is available from truncated lognormal moments."
        ),
        h2("4.1 Truncated lognormal moment identity"),
        p("If X~N(mu,s^2), then for any real a and threshold b:"),
        eq(
            """
E[ exp(aX) 1_{X <= b} ]
= exp(a mu + 0.5 a^2 s^2)
  Phi((b-mu-a s^2)/s).
"""
        ),
        p(
            "Derivation: multiply the normal density by exp(aX), complete the square, "
            "and recognize a shifted normal density. Differences of this expression over "
            "two thresholds give interval moments. These moments produce exact means and "
            "variances for clip(exp(X)-1, local_floor, local_cap)."
        ),
        h2("4.2 Benchmark independence"),
        p(
            "Training and benchmark random streams are independent. A proxy should not "
            "be evaluated against the same paths used to train it; common random numbers "
            "may make a poor proxy appear artificially accurate."
        ),
        PageBreak(),
    ]
)

# Basis and transforms
story.extend(
    [
        h1("5. Proxy targets and regularized bases"),
        h2("5.1 Coordinate scaling"),
        eq(
            """
x_scaled = 2 (x-x_min)/(x_max-x_min) - 1.
"""
        ),
        p(
            "Chebyshev and Bernstein systems are evaluated on compact intervals. Inputs "
            "outside the trained domain are clipped in these experiments; production "
            "systems should define explicit extrapolation policies."
        ),
        h2("5.2 Ridge regression"),
        eq(
            """
min_beta ||A beta - y||_2^2 + lambda ||Gamma beta||_2^2

Normal equation:
(A'A + lambda Gamma'Gamma) beta = A'y.
"""
        ),
        p(
            "The intercept is not penalized. Ridge stabilizes noisy Monte Carlo labels and "
            "ill-conditioned high-order bases."
        ),
        h2("5.3 Chebyshev basis"),
        eq(
            """
T_0(x)=1, T_1(x)=x,
T_(n+1)(x)=2x T_n(x)-T_(n-1)(x).

Proxy: y_hat(x)=Sum_(k=0)^d beta_k T_k(x).
"""
        ),
        p(
            "Chebyshev polynomials are better conditioned on [-1,1] than monomials and "
            "work well for globally smooth value functions."
        ),
        PageBreak(),
        h2("5.3.1 Tensor basis and sparse multi-index set"),
        p(
            "For scaled state z=(z_1,...,z_D), a multivariate tensor Chebyshev term is"
        ),
        eq(
            """
Psi_alpha(z) = Product_(j=1)^D T_(alpha_j)(z_j),
alpha = (alpha_1,...,alpha_D), alpha_j >= 0.

Full tensor degree p: 0 <= alpha_j <= p
Number of terms: (p+1)^D.
"""
        ),
        p(
            "The full tensor is impossible in seven dimensions. Sparse regression keeps "
            "only a payoff-aware index set. A standard anisotropic form is"
        ),
        eq(
            """
I_p = { alpha : Sum_j w_j alpha_j <= p },

where small w_j permits higher degree in an important coordinate and large w_j
suppresses degree in a weak coordinate.
"""
        ),
        h2("5.3.2 Basket cliquet sparse basis"),
        p(
            "The basket implementation gives the lower and upper payoff cushions the "
            "largest degree because they locate the global floor and cap transitions. "
            "The six spot/variance coordinates enter through lower-order interactions:"
        ),
        eq(
            """
1. T_a(lower cushion),                 a=0,...,15
2. T_b(upper cushion),                 b=1,...,8
3. T_a(lower) T_b(upper),              a+b<9
4. T_a(lower) state_j,                 a=0,...,5
5. T_b(upper) state_j,                 b=0,...,3
6. state_j state_k,                    j<k.
"""
        ),
        p(
            "This is not a generic total-degree polynomial. It encodes the belief that "
            "payoff-boundary location is strongly nonlinear, while residual SLV state "
            "dependence is smoother and lower order."
        ),
        h2("5.3.3 Sparse Chebyshev training algorithm"),
        eq(
            """
Input: states x_i, MC values V_i, known bounds L<U.

1. Scale each feature to z_ij in [-1,1].
2. Transform p_i=clip((V_i-L)/(U-L),eps,1-eps).
3. Set y_i=log(p_i/(1-p_i)).
4. Build A_(i,k)=Psi_(alpha_k)(z_i) for selected terms.
5. Solve (A'A + lambda Gamma) beta = A'y.
6. At query x, evaluate y_hat=Psi(x)' beta.
7. Return V_hat=L+(U-L)/(1+exp(-y_hat)).
8. Override exact floor, cap, or maturity states.
"""
        ),
        p(
            "With M retained terms and N labels, evaluation is O(M) per state. Normal-"
            "equation training is approximately O(N M^2 + M^3), but it is convex, "
            "deterministic, and normally much cheaper than path generation."
        ),
        PageBreak(),
        h2("5.4 Bernstein and global Bezier basis"),
        eq(
            """
B_(k,n)(u) = C(n,k) u^k (1-u)^(n-k),  0<=u<=1.
y_hat(u) = Sum_(k=0)^n c_k B_(k,n)(u).
"""
        ),
        p(
            "The Bernstein basis is nonnegative and sums to one. With fitted coefficients "
            "c_k it is a global Bezier curve. It provided the best European 1D result."
        ),
        h2("5.5 Log and bounded-logit targets"),
        eq(
            """
Positive value: y = log(V + epsilon), V_hat=exp(y_hat)-epsilon.

Known bounds L<U:
p=(V-L)/(U-L), y=log(p/(1-p)),
V_hat=L+(U-L)/(1+exp(-y_hat)).
"""
        ),
        p(
            "The log target prioritizes relative accuracy. The logit target guarantees "
            "hard global floor/cap bounds for cliquets."
        ),
        PageBreak(),
    ]
)

# Interpolants
story.extend(
    [
        h1("6. PCHIP, Akima, and piecewise Bezier curves"),
        h2("6.1 Cubic Hermite segment"),
        eq(
            """
t=(x-x_i)/h, h=x_(i+1)-x_i

H(t)=h00(t)y_i + h10(t)h d_i
     + h01(t)y_(i+1) + h11(t)h d_(i+1)

h00=2t^3-3t^2+1, h10=t^3-2t^2+t
h01=-2t^3+3t^2, h11=t^3-t^2.
"""
        ),
        h2("6.2 PCHIP slope rule"),
        p(
            "Let secants delta_i=(y_(i+1)-y_i)/h_i. If adjacent secants have opposite "
            "sign, PCHIP sets the interior derivative to zero. Otherwise it uses a "
            "weighted harmonic mean:"
        ),
        eq(
            """
d_i = (w1+w2)/(w1/delta_(i-1)+w2/delta_i)
w1=2h_i+h_(i-1), w2=h_i+2h_(i-1).
"""
        ),
        p(
            "This suppresses overshoot and preserves monotonicity. It is especially useful "
            "for American continuation curves near an exercise boundary."
        ),
        PageBreak(),
        h2("6.2.1 Endpoint slopes and limiting"),
        p(
            "Interior slopes use information on both sides. At the left endpoint only the "
            "first two secants are available, so PCHIP first forms a one-sided estimate:"
        ),
        eq(
            """
d_0_raw = [(2h_0+h_1) delta_0 - h_0 delta_1] / (h_0+h_1).
"""
        ),
        p(
            "Then it applies two safety rules. If d_0_raw has the wrong sign relative to "
            "delta_0, set d_0=0. If its magnitude exceeds 3|delta_0|, set "
            "d_0=3 delta_0. The right endpoint uses the reflected formula."
        ),
        h2("6.2.2 Why the harmonic mean is shape preserving"),
        p(
            "Suppose two neighboring secants are positive. Their weighted harmonic mean "
            "is also positive and cannot exceed the larger secant by an arbitrary amount. "
            "If the secants disagree in sign, the data have a local turning point; setting "
            "the derivative to zero prevents the cubic from inventing an extra turn. The "
            "formal monotonicity conditions and proof are given in Appendix F."
        ),
        h2("6.2.3 PCHIP implementation recipe"),
        eq(
            """
Build once:
1. Sort pairs (x_i,y_i) and remove duplicate x_i.
2. Compute h_i=x_(i+1)-x_i and delta_i=(y_(i+1)-y_i)/h_i.
3. Compute interior d_i with the sign test and harmonic mean.
4. Compute and limit the two endpoint slopes.

Evaluate at x:
5. Find i such that x_i <= x <= x_(i+1).
6. Set t=(x-x_i)/h_i.
7. Evaluate the four cubic Hermite basis functions.
8. Return H_i(t), then invert any log or logit target transform.
9. Override exact payoff, floor, cap, or exercise regions.
"""
        ),
        p(
            "Building the curve costs O(n). With binary search, each query costs O(log n); "
            "on an ordered query grid the interval index can be advanced in O(1) amortized "
            "time. PCHIP interpolates every label, so it must be paired with low-noise MC "
            "labels; it is a shape-preserving interpolator, not a denoising method."
        ),
        h2("6.3 Akima slope rule"),
        eq(
            """
d_i = [w_left delta_(i-1) + w_right delta_i]
      / (w_left+w_right)

w_left  = |delta_(i+1)-delta_i|
w_right = |delta_(i-1)-delta_(i-2)|.
"""
        ),
        p(
            "Akima weights the less rapidly changing side more heavily. It is local and "
            "less aggressively monotone than PCHIP. This matched the reduced Asian curve "
            "very well."
        ),
        h2("6.4 Bezier/Hermite equivalence"),
        eq(
            """
For the same endpoint slopes:
P0=y_i
P1=y_i + h d_i/3
P2=y_(i+1) - h d_(i+1)/3
P3=y_(i+1).

B(t)=(1-t)^3 P0 + 3(1-t)^2 t P1
     +3(1-t)t^2 P2 + t^3 P3.
"""
        ),
        p(
            "Expanding B(t) gives the Hermite polynomial exactly. Therefore a piecewise "
            "Bezier curve built from PCHIP slopes is PCHIP in another representation, "
            "not a distinct smoother."
        ),
        PageBreak(),
        h2("6.5 Expanded 99-case smoother study"),
        p(
            "Seventeen distinct one-dimensional estimators were tested over European, "
            "American, Asian, and barrier parameter combinations. Every fitted surface "
            "used a 10M-scenario training budget; MC benchmarks used 500,000 paths per "
            "state. Product families received equal aggregate weight."
        ),
        table(
            [
                ["Method", "Balanced p99", "Max for V>=0.05", "Local overshoot"],
                ["Natural cubic interpolation", "1.799%", "1.163%", "2.01%"],
                ["PCHIP", "1.834%", "1.220%", "0.00%"],
                ["Akima", "1.958%", "1.286%", "0.17%"],
                ["MAKIMA", "1.968%", "1.317%", "0.15%"],
                ["Cubic smoothing spline", "2.098%", "1.440%", "2.49%"],
                ["Linear interpolation", "2.683%", "2.058%", "0.00%"],
            ],
            widths=[2.4 * inch, 1.35 * inch, 1.55 * inch, 1.35 * inch],
        ),
        p(
            "Natural cubic interpolation led average accuracy by only 0.035 percentage "
            "points. PCHIP won 53 of 99 head-to-head p99 cases and introduced no local "
            "overshoot, so PCHIP remains the generic one-feature PFE default."
        ),
        PageBreak(),
    ]
)

# European
story.extend(
    [
        h1("7. European option proxy"),
        h2("7.1 State and benchmark"),
        p(
            "For constant-parameter GBM, the state at a fixed time is spot alone. The "
            "benchmark is the Black-Scholes formula. For a call:"
        ),
        eq(
            """
d1=[log(S/K)+(r-q+sigma^2/2)tau]/(sigma sqrt(tau))
d2=d1-sigma sqrt(tau)
C=S exp(-q tau) Phi(d1)-K exp(-r tau) Phi(d2).
"""
        ),
        h2("7.2 Why d1 is the fitting coordinate"),
        p(
            "d1 combines spot, strike, volatility, carry, and remaining maturity into the "
            "standardized coordinate controlling exercise probability and delta. Sampling "
            "uniformly in d1 is equivalent to deliberate wing coverage in delta space."
        ),
        h2("7.3 Training label"),
        p(
            "For each d1 state, shifted antithetic terminal normals produce an unbiased MC "
            "label. The default target is log(value), originally fitted by degree-7 "
            "Chebyshev ridge."
        ),
        h2("7.4 One-dimensional method comparison"),
        table(
            [
                ["Method", "Worst max error", "Average p99", "Average MAE"],
                ["Global Bernstein/Bezier", "0.536%", "0.208%", "0.003546"],
                ["Chebyshev", "0.854%", "0.326%", "0.004423"],
                ["Akima", "2.540%", "1.127%", "0.007015"],
                ["PCHIP", "2.554%", "1.149%", "0.007098"],
                ["Bezier with PCHIP slopes", "2.554%", "1.149%", "0.007098"],
            ],
            widths=[2.5 * inch, 1.35 * inch, 1.35 * inch, 1.35 * inch],
        ),
        p(
            "Conclusion: the European curve is sufficiently smooth that global regularized "
            "bases average MC noise better than exact local interpolation."
        ),
        p(
            "For implementation consistency, the standalone universal default is now "
            "log-PCHIP in d1. Its independent-seed worst error is 2.637%, versus the "
            "product-specific Bernstein result of 0.536%."
        ),
        PageBreak(),
    ]
)

# Asian derivation
story.extend(
    [
        h1("8. Arithmetic Asian option proxy"),
        h2("8.1 Raw Markov state"),
        p(
            "At fixing index j, store current spot S and running sum A of fixings strictly "
            "before today. Let N be total fixings and m=N-j-1 future fixings."
        ),
        eq(
            """
Payoff = [ (A + S + Sum_(k=1)^m S_(j+k))/N - K ]_+.
"""
        ),
        h2("8.2 Adjusted-strike derivation"),
        eq(
            """
Payoff
= (m/N) [ (1/m) Sum_(k=1)^m S_(j+k) - K_adj ]_+

K_adj = (N K - A - S)/m.
"""
        ),
        p(
            "This is an algebraic identity. It rotates the diagonal payoff boundary in "
            "(S,A) into a strike-like scalar."
        ),
        h2("8.3 GBM homogeneity reduction"),
        p(
            "Future GBM spots satisfy S_(j+k)=S G_k, where the joint growth vector G does "
            "not depend on S. Therefore:"
        ),
        eq(
            """
V(S,A,j)
= (m/N) E[ (S average(G)-K_adj)_+ ]
= S (m/N) E[ (average(G)-K_adj/S)_+ ].
"""
        ),
        p(
            "At a fixed date the normalized value V/S is a one-dimensional function of "
            "K_adj/S, represented by an adjusted d1-like coordinate. This is why a generic "
            "2D fit is unnecessary for this GBM product."
        ),
        h2("8.4 Exact linear wing"),
        p(
            "If K_adj<=0 for a call, every future arithmetic average is positive relative "
            "to the adjusted strike, so the positive-part operator can be removed. The "
            "value is then the discounted expected average minus strike and is computed "
            "analytically."
        ),
        PageBreak(),
    ]
)

story.extend(
    [
        h1("9. Asian benchmark and fitting results"),
        h2("9.1 Geometric Asian control"),
        p(
            "The log of the discrete future geometric average is a weighted sum of normal "
            "increments, hence normal. If weights are w_l=(m-l+1)/m:"
        ),
        eq(
            """
log(G_geo) = log(S)
 + (r-q-sigma^2/2) dt (m+1)/2
 + sigma sqrt(dt) Sum_l w_l Z_l.

Var(log(G_geo)) = sigma^2 dt Sum_l w_l^2.
"""
        ),
        p(
            "Thus the geometric Asian call has a Black-Scholes-like exact value and is a "
            "highly correlated control for the arithmetic payoff."
        ),
        h2("9.2 Hybrid target"),
        p(
            "Fit log(V/S) in the OTM and moderate region. In the deep ITM region fit "
            "log((V-linear_baseline)/S), then add the exact baseline. This prevents a "
            "large intrinsic component from hiding time-value error."
        ),
        h2("9.3 1D fitter comparison"),
        table(
            [
                ["Method", "Worst max error", "Average p99", "Average MAE"],
                ["Akima", "0.186%", "0.057%", "0.000178"],
                ["Chebyshev", "3.286%", "1.721%", "0.013523"],
                ["PCHIP", "3.453%", "0.936%", "0.000273"],
                ["Global Bernstein/Bezier", "6.854%", "3.303%", "0.017907"],
                ["Bezier with PCHIP slopes", "3.453%", "0.936%", "0.000273"],
            ],
            widths=[2.5 * inch, 1.35 * inch, 1.35 * inch, 1.35 * inch],
        ),
        p(
            "Akima is the preferred 1D fitter for this reduced Asian experiment. PCHIP is "
            "usable, but its monotonicity limiter can flatten local derivatives when MC "
            "labels contain small slope reversals. A second independent-seed run confirmed "
            "Akima at 0.144% worst error, versus 2.577% for Chebyshev and 1.643% for PCHIP."
        ),
        p(
            "The standalone universal default is nevertheless PCHIP. With its independent "
            "default seed it achieved 3.331% worst error while sharing the same fitting "
            "engine as European and American options."
        ),
    ]
)
add_plot(
    story,
    "AsianOptExperiment/results/plots/asian_day_06_hybrid.png",
    "Asian adjusted-moneyness proxy diagnostic from the broader experiment.",
)
story.append(PageBreak())

# Basket Asian
story.extend(
    [
        h1("9A. Ten-asset basket Asian extension"),
        h2("9A.1 Contract and Markov state"),
        p(
            "The basket Asian experiment prices a monthly arithmetic Asian call on an "
            "equal-weight basket of ten correlated GBM assets. Some pairwise correlations "
            "are positive and some are negative. At fixing index j the state is the ten-"
            "dimensional spot vector S=(S_1,...,S_10) plus the running sum B of previous "
            "basket fixings."
        ),
        eq(
            """
b(S) = (S_1+...+S_10)/10

Payoff = [ (B + b(S) + Sum_(k=1)^m b(S_(j+k)))/N - K ]_+.
"""
        ),
        p(
            "Unlike the single-asset GBM Asian, this cannot be reduced exactly to one "
            "adjusted-moneyness coordinate because the ten assets have different "
            "volatilities, dividends, and correlations. The feature design therefore "
            "uses moment information plus PCA directions."
        ),
        h2("9A.2 Moment baseline"),
        p(
            "For each state the code computes the first two moments of the final "
            "arithmetic basket average exactly under correlated GBM. The identity used is"
        ),
        eq(
            """
E[S_i(t_a) S_k(t_b)]
= S_i S_k exp(mu_i t_a + mu_k t_b
              + rho_ik sigma_i sigma_k min(t_a,t_b)).
"""
        ),
        p(
            "Those moments define a moment-matched lognormal approximation. By itself it "
            "is smooth and fast, but its day-0 worst max error was 28.450% on the "
            "524,288-path benchmark, so it is used as a baseline rather than the final "
            "proxy."
        ),
        h2("9A.3 PCA sparse-Chebyshev correction"),
        p(
            "The correction features are log expected-average moneyness, effective "
            "average volatility, current basket moneyness, running-average moneyness, "
            "cross-sectional dispersion, four principal-component scores of log spots, "
            "and log(1+baseline value). A sparse Chebyshev ridge model is fitted to a "
            "bounded log-factor correction around the moment baseline."
        ),
        p(
            "The final default adds a one-dimensional PCHIP calibration of the training "
            "residual as a function of expected-average moneyness. This is not an ad hoc "
            "hand correction: it is a second-stage smoother trained only on Monte Carlo "
            "labels, and it addresses the common situation where a global high-dimensional "
            "basis has small but systematic tail bias."
        ),
        h2("9A.4 Basket Asian result"),
        table(
            [
                ["Method", "Worst max error", "Average p99", "Average MAE"],
                ["Moment lognormal baseline", "28.450%", "5.947%", "0.023189"],
                ["Relative residual sparse Chebyshev", "13.200%", "2.496%", "0.018653"],
                ["Log-factor sparse Chebyshev", "8.995%", "1.996%", "0.012734"],
                ["PCHIP-calibrated log factor", "5.873%", "1.820%", "0.009023"],
                ["Raw residual sparse Chebyshev", "19.085%", "3.339%", "0.002380"],
                ["Fixed residual/log-factor blend", "11.452%", "2.295%", "0.006251"],
            ],
            widths=[2.55 * inch, 1.35 * inch, 1.35 * inch, 1.35 * inch],
        ),
        p(
            "The default therefore meets the requested under-8% worst-error target on "
            "the tested state design. Training uses 513 states per date and 65,536 Sobol "
            "or Sobol/LR paths per state, or about 33.6 million state-scenarios per date "
            "after the power-of-two Sobol rounding. Each validation state uses 524,288 "
            "benchmark paths."
        ),
        p(
            "Importance-sampling note: the current script uses a two-component Gaussian "
            "likelihood-ratio mixture for the long-dated day-0 slice, shifting basket "
            "growth toward the exercise boundary and weighting each path by p/q. It uses "
            "plain Sobol for later dates because the same shift increased weighted "
            "variance there."
        ),
    ]
)
add_plot(
    story,
    "BasketAsianOptExperiment/results/plots/basket_asian_day_00_pchip_calibrated_log_factor_pca.png",
    "Ten-asset basket Asian day-0 diagnostic using the PCHIP-calibrated sparse proxy.",
)
story.append(PageBreak())

# American
story.extend(
    [
        h1("10. American put: optimal stopping"),
        h2("10.1 Snell envelope and Bellman recursion"),
        p(
            "For exercise dates t_k, intrinsic payoff g(S)=(K-S)_+. The value is the "
            "smallest supermartingale dominating g, equivalently the Snell envelope:"
        ),
        eq(
            """
V_M(S)=g(S)
C_k(S)=exp(-r dt) E[ V_(k+1)(S_next) | S_k=S ]
V_k(S)=max(g(S), C_k(S)).
"""
        ),
        p(
            "The max creates a moving derivative kink at the exercise boundary. A global "
            "polynomial fitted recursively can turn a small local continuation error into "
            "a systematic bias at every earlier date."
        ),
        h2("10.2 MC dynamic programming"),
        p(
            "At each of 100 exercise dates, the implementation uses 121 log-spaced state "
            "nodes, antithetic one-step GBM transitions, and about 10 million transitions "
            "over the complete backward pass. It fits C_k, then imposes max(g,C_k) exactly."
        ),
        h2("10.3 Why PCHIP wins here"),
        p(
            "PCHIP is local, shape preserving, and does not ring around the moving boundary. "
            "The broad experiment achieved 2.249% worst max error; an independent seed "
            "achieved 5.289%. Global log-Chebyshev fits were unstable because exponentiated "
            "errors recursively compounded."
        ),
        p(
            "Akima was also tested through the complete 100-date recursion. It achieved "
            "3.225% worst max error, 1.675% average p99, and 0.002033 average MAE, versus "
            "2.249%, 0.947%, and 0.001479 for PCHIP. Akima is viable, but PCHIP's "
            "monotonicity-preserving slopes are safer near the exercise boundary."
        ),
        h2("10.4 Independent benchmark"),
        p(
            "A projected implicit finite-difference solver uses 4,000 time steps and 2,000 "
            "spot steps. Projected SOR enforces V>=g. At S=100, t=0, the finite-difference "
            "value is about 6.6597 versus 6.6605 from a separate 4,000-step CRR tree."
        ),
    ]
)
add_plot(
    story,
    "AmericanOptExperiment/results/plots/american_step_040_pchip_spline.png",
    "American put value, signed error, and signed relative error at time 0.4.",
)
story.append(PageBreak())

# Barrier options
story.extend(
    [
        h1("10A. Single and double barrier options"),
        h2("10A.1 State and variants"),
        p(
            "The experiment prices zero-rebate down-and-out, up-and-out, and "
            "double-knock-out calls with either monthly discrete or continuous "
            "monitoring. Conditional on no historical hit, spot is the only fitted "
            "feature. A live/hit indicator must be added once monitoring history exists."
        ),
        p(
            "Zero-rebate knock-ins require no second MC training run. Under the same "
            "monitoring convention, in/out parity gives V_in = V_vanilla - V_out."
        ),
        h2("10A.2 Exact Brownian-bridge segment correction"),
        p(
            "For log endpoints x and y above lower log barrier b over a GBM step with "
            "variance q=sigma^2 dt, reflection gives the conditional survival probability"
        ),
        eq(
            """
P(min bridge > b | x,y)
  = 1 - exp[-2 (x-b)(y-b)/q],  x>b, y>b.
"""
        ),
        p(
            "The upper-barrier expression replaces the distances by b-x and b-y. "
            "Multiplying segment survival probabilities and weighting the payoff "
            "integrates out crossing indicators, reducing variance. For a double "
            "barrier, the absorbing interval transition density is evaluated by a "
            "method-of-images series and divided by the free transition density."
        ),
        h2("10A.3 Sampling and results"),
        p(
            "Log-Chebyshev spot nodes cover the alive domain. Common random numbers make "
            "label noise smooth across spot, antithetics reduce path noise, and shifted "
            "interior validation nodes prevent training-grid leakage."
        ),
        table(
            [
                ["Variant", "PCHIP raw worst", "Worst for V>=0.05"],
                ["Down-out discrete", "2.253%", "2.087%"],
                ["Down-out continuous", "2.531%", "1.762%"],
                ["Up-out discrete", "9.096%", "4.853%"],
                ["Up-out continuous", "8.808%", "5.074%"],
                ["Double-out discrete", "8.387%", "4.999%"],
                ["Double-out continuous", "8.148%", "4.916%"],
            ],
            widths=[2.5 * inch, 1.8 * inch, 2.0 * inch],
        ),
        p(
            "The 8-9% raw maxima are one- or two-cent option values. Meaningful-value "
            "errors remain around 2% for down barriers and about 5% for up/double "
            "barriers. PCHIP remains the default because the broader 99-case study found "
            "near-best accuracy with zero local overshoot."
        ),
    ]
)
add_plot(
    story,
    "BarrierOptExperiment/results/plots/double_out_continuous_month_06_pchip.png",
    "Continuous double-knock-out value, signed error, and signed relative error.",
)
story.append(PageBreak())

# Cliquet
story.extend(
    [
        h1("11. Single-name GBM cliquet"),
        h2("11.1 Payoff"),
        eq(
            """
c_i = clip(S_i/S_(i-1)-1, local_floor, local_cap)
H = notional * clip(Sum_i c_i, global_floor, global_cap).
"""
        ),
        h2("11.2 State reduction theorem"),
        p(
            "Under GBM, each future reset return S_i/S_(i-1)-1 depends only on the new "
            "normal increment and fixed model parameters, not on S_(i-1). Future coupons "
            "are therefore independent of current spot conditional on reset time."
        ),
        eq(
            """
State at reset = accrued = Sum of realized clipped coupons.
V = V(accrued, remaining_periods).
"""
        ),
        p(
            "This is an exact scale-invariance reduction, not an empirical approximation."
        ),
        h2("11.3 Expected-total coordinate"),
        eq(
            """
z = [accrued + m E(c) - midpoint(global bounds)]
    / sqrt(m Var(c)).
"""
        ),
        p(
            "A boundary-enriched grid samples around the estimated floor and cap transitions. "
            "Known flat tails are exact: if even all remaining local caps cannot leave the "
            "global floor, or all local floors force the cap, regression is bypassed."
        ),
        h2("11.4 Result"),
        p(
            "Bounded-logit Chebyshev degree 19 achieved 3.593% worst max error, 0.928% "
            "average p99, and 0.003194 average MAE."
        ),
    ]
)
add_plot(
    story,
    "CliquetOptExperiment/results/plots/cliquet_day_09_logit_z_boundary_d19.png",
    "GBM cliquet proxy diagnostic with three coupons remaining.",
)
story.append(PageBreak())

# SLV cliquet
story.extend(
    [
        h1("12. Single-name SLV cliquet"),
        h2("12.1 Expanded state"),
        p(
            "SLV destroys GBM scale invariance because leverage L(S) depends on spot and "
            "future variance depends on current v. The reset state becomes "
            "(accrued, S, v)."
        ),
        h2("12.2 Structured features"),
        p(
            "A frozen-coefficient one-period coupon distribution supplies conditional mean "
            "and variance. These form an expected-total coordinate z, supplemented by "
            "log(S/S0) and log(v/theta). The target remains bounded logit."
        ),
        h2("12.3 Maturity-adaptive fit"),
        table(
            [
                ["Remaining coupons", "Fitter", "Reason"],
                ["9-12", "Local quadratic", "Broad smooth long-horizon dependence"],
                ["3-6", "Anisotropic Chebyshev", "Sharper payoff transitions"],
                ["0", "Exact payoff", "No continuation uncertainty"],
            ],
            widths=[1.45 * inch, 2.0 * inch, 3.35 * inch],
        ),
        p(
            "Lower-tail mixture importance sampling was necessary for seed stability. The "
            "common-random-number rerun achieved 5.067% worst error. A separate two-design "
            "study rebuilt 10M-path labels and used 500,000 benchmark paths per state on "
            "both designs; the fixed adaptive hybrid achieved 4.795% worst error."
        ),
        p(
            "Anisotropic Chebyshev degrees 13-23, sparse Hermite, local regression, and "
            "Nystrom Matern regression were tested in that study. None beat the existing "
            "time-based local/spectral rule. Common random numbers are retained because "
            "they smooth label noise across state without changing conditional means."
        ),
    ]
)
add_plot(
    story,
    "SLVCliquetOptExperiment/results/plots/slv_cliquet_day_09_adaptive_hybrid.png",
    "Single-name SLV cliquet validation states ranked by benchmark value.",
)
story.append(PageBreak())

# Basket
story.extend(
    [
        h1("13. Three-underlying SLV basket cliquet"),
        h2("13.1 Generalized coupon definitions"),
        table(
            [
                ["Variant", "Monthly coupon"],
                ["Basket return", "clip(mean(R1,R2,R3), local floor, local cap)"],
                ["Weighted average", "clip(sum_i w_i R_i, local floor, local cap)"],
                ["Basket ratio", "clip(sum_i w_i S_i(end) / sum_i w_i S_i(start) - 1, local floor, local cap)"],
                ["Average clipped", "mean(clip(R1),clip(R2),clip(R3))"],
                ["Second worst", "clip(second order statistic of R_i, local floor, local cap)"],
                ["Worst of", "clip(min(R1,R2,R3), local floor, local cap)"],
                ["Best of", "clip(max(R1,R2,R3), local floor, local cap)"],
                ["Spread bonus", "clip(weighted average) - L_spd clip(max R_i - min R_i) + bonus 1_{basket ratio >= 0}"],
            ],
            widths=[1.55 * inch, 5.25 * inch],
        ),
        eq(
            """
R_i^(j) = S_i(T_j^end) / S_i(T_j^start) - 1

R_basket^(j) = [sum_i w_i S_i(T_j^end)] / [sum_i w_i S_i(T_j^start)] - 1

C_j = local coupon after clipping, spread adjustment, and optional bonus
G = sum_(j=1)^m C_j
Payoff = N clip(G, global floor, global cap).
"""
        ),
        h2("13.2 Seven-dimensional Markov state"),
        eq(
            """
X = (accrued, S1,S2,S3, v1,v2,v3).
"""
        ),
        p(
            "Each asset has its own variance factor and leverage function. Correlated market "
            "drivers create cross-asset dependence; each spot shock also contains its own "
            "negatively correlated variance shock."
        ),
        eq(
            """
Z_i^S = rho_i Z_i^v + sqrt(1-rho_i^2) Z_i^M,
Corr(Z^M)=R_market.
"""
        ),
        p(
            "This construction is automatically positive semidefinite because it is a "
            "linear transformation of independent variance normals and a valid correlated "
            "market-normal vector."
        ),
        h2("13.3 Sampling"),
        p(
            "The current generalized experiment uses 1,009 low-discrepancy market states "
            "and 17 accrued-return layers per market state. The labels are grouped: one "
            "Sobol SLV simulation produces future coupon sums, and those same sums price "
            "all accrued layers for that market state. This spends the path budget on "
            "future uncertainty rather than resimulating identical spot/variance states."
        ),
        p(
            "Each market-state label uses 32,768 Sobol paths after the component-count "
            "rounding. Validation uses 31 independent states and 524,288 Sobol paths per "
            "state. The path sampler is a likelihood-ratio mixture over common market "
            "directions and dispersion directions, with antithetic pairs inside each "
            "component."
        ),
        eq(
            """
q(z) = (1/K) sum_(k=1)^K phi(z - mu_k)

weight(z) = phi(z) / q(z)
          = 1 / [(1/K) sum_k exp(mu_k' z - ||mu_k||^2 / 2)].
"""
        ),
        h2("13.4 PCA and payoff-aware coordinates"),
        p(
            "Raw spot and variance coordinates are retained, but the proxy also uses "
            "orthogonal PCA directions from the instantaneous covariance matrix. In three "
            "dimensions these are market level plus two spread-like modes. The feature "
            "set also contains lower/upper payoff cushions, coupon skewness, local floor "
            "and cap masses, weighted averages, minima, maxima, and dispersion measures."
        ),
        eq(
            """
Sigma_0 = diag(sqrt(theta)) R_market diag(sqrt(theta))
Sigma_0 e_k = lambda_k e_k

spot PCA features     = log(S/S0)' e_k
variance PCA features = log(v/theta)' e_k.
"""
        ),
        h2("13.5 Result and limitation"),
        p(
            "Grouped labels and PCA features materially improved the basket-like coupons, "
            "but they did not solve all generalized coupon styles. Order-statistic coupons "
            "remain hard because a small value near the global floor is a rare event in a "
            "bounded sum of clipped maximum or minimum returns. The result is useful but "
            "not yet a universal 5-8% basket-cliquet proxy."
        ),
        table(
            [
                ["Variant", "Best method", "Worst max", "Average p99"],
                ["Basket return", "Adaptive local/sparse blend", "6.448%", "3.060%"],
                ["Weighted average", "Adaptive local/sparse blend", "11.280%", "4.946%"],
                ["Basket ratio", "Adaptive local/sparse blend", "19.942%", "7.946%"],
                ["Average clipped", "Adaptive local/sparse blend", "7.967%", "4.838%"],
                ["Second worst", "Local summary quadratic", "11.326%", "7.078%"],
                ["Worst of", "Local summary quadratic", "29.002%", "13.103%"],
                ["Best of", "Local summary quadratic", "66.880%", "24.321%"],
                ["Spread bonus", "Local summary quadratic", "15.911%", "12.466%"],
            ],
            widths=[1.45 * inch, 2.35 * inch, 1.25 * inch, 1.25 * inch],
        ),
        PageBreak(),
    ]
)

story.extend(
    [
        h1("14. Basket high-dimensional model search"),
        p(
            "The search compared local full-state and summary regressions, sparse anisotropic "
            "Chebyshev terms, Gaussian RBF kernels, a two-layer tanh neural ensemble, "
            "moment-anchored residual fits, accrued-return PCHIP/kNN interpolation, and "
            "payoff-aware PCA features. With these training sizes, the robust winners are "
            "still simple local/sparse regressions for most variants."
        ),
        h2("14.1 Grouped-label improvement"),
        eq(
            """
For fixed market state M and simulated future coupon sums C_path:

V(a, M) = discount E[ N clip(a + C_path, floor, cap) | M ].

Thus one simulation of C_path can price many accrued values a_1,...,a_L.
"""
        ),
        p(
            "This grouped construction is exact under the same simulated future paths; it "
            "does not approximate the payoff. It simply reuses the path sums for multiple "
            "initial accrued values. It reduced training noise substantially compared with "
            "simulating every accrued layer independently."
        ),
        h2("14.2 Why the order-statistic cases remain difficult"),
        p(
            "For best-of and worst-of coupons, the future sum distribution is bounded, "
            "skewed, and sensitive to cross-asset dispersion. Near the global floor the "
            "value may be only a few cents, so a small absolute miss creates a large "
            "percentage error. However, some misses are still too large to dismiss: the "
            "month-6 best-of case has a 0.048 dollar miss on a 0.076 dollar benchmark."
        ),
        table(
            [
                ["Method idea", "Outcome"],
                ["Moment-normal anchor", "Too Gaussian; overestimates bounded rare tails"],
                ["Anchored sparse Chebyshev", "Reduced neither max error nor p99"],
                ["Accrued PCHIP/kNN", "Good structure, but market-state interpolation overpredicted tails"],
                ["PCA/spread features", "Helpful as diagnostics, not sufficient alone"],
                ["Grouped accrued labels", "Clear improvement; kept as methodology"],
            ],
            widths=[2.2 * inch, 4.4 * inch],
        ),
        p(
            "The next credible improvement is adaptive state enrichment targeted at the "
            "failed order-statistic neighborhoods, or a larger path-level neural model "
            "trained directly on simulated paths. A fixed generic spline or polynomial "
            "basis is probably not enough for every generalized basket cliquet style."
        ),
        h2("14.3 Literature-inspired residual search"),
        p(
            "A clipped-normal moment baseline was used as a low-fidelity model, followed "
            "by direct or residual sparse Hermite, local, Nystrom Matern, and fixed "
            "ensemble corrections. Coupon skewness and floor/cap masses were added as "
            "features. Inverse-variance weighted sparse Chebyshev fits were also tested "
            "on both 2,001 and 5,001 states."
        ),
        p(
            "None improved worst-case relative error on the development and independent "
            "designs. Some dense weighted fits reduced MAE but leaked larger errors into "
            "tail states. The moment baseline did not simplify correlated clipped and "
            "worst-of SLV coupons enough for residual learning to help."
        ),
    ]
)
add_plot(
    story,
    "BasketCliquetOptExperiment/results/plots/basket_return_month_09_sparse_chebyshev.png",
    "Development diagnostic for the fixed sparse basket-return proxy with three coupons remaining.",
)
story.append(PageBreak())

# Generic methodology
story.extend(
    [
        h1("15. Generic workflow for a new instrument"),
        h2("Step 1: define the contract exactly"),
        bullet("Observation schedule, exercise rights, local/global bounds, and settlement."),
        bullet("Clarify basket aggregation order: clip-then-average is not average-then-clip."),
        h2("Step 2: identify a sufficient Markov state"),
        bullet("Start from the simulator state and add path statistics required by payoff."),
        bullet("Prove any dimension reduction from scale invariance or payoff algebra."),
        h2("Step 3: nondimensionalize"),
        bullet("Use moneyness, normalized variance, expected-total cushions, and time."),
        bullet("Align transition boundaries across maturities before fitting."),
        h2("Step 4: exploit exact regions"),
        bullet("Terminal payoff, linear ITM wings, hard global floor/cap tails, intrinsic value."),
        h2("Step 5: design state sampling"),
        bullet("Combine broad low-discrepancy coverage with boundary-focused clusters."),
        bullet("Do not spend most paths in flat regions that are already known analytically."),
        h2("Step 6: generate MC labels"),
        bullet("Use antithetic paths by default."),
        bullet("Add controls when exact expectations are available."),
        bullet("Attempt likelihood-ratio shifts for rare payoff wings whenever p/q is known and numerically stable."),
        bullet("Keep the shift only on slices where the weighted estimator lowers variance or tail error."),
        bullet("Keep state-space enrichment distinct from true importance sampling in the run notes."),
        h2("Step 7: choose fitter by effective dimension"),
        table(
            [
                ["Effective dimension", "Default candidates"],
                ["Any genuine 1D price proxy", "PCHIP operational default"],
                ["1D smooth, product optimized", "Natural cubic, Chebyshev, Bernstein"],
                ["1D noisy labels", "Smoothing spline or P-spline after validation"],
                ["2-5D smooth", "Sparse anisotropic basis, local polynomial, small NN"],
                ["6D+", "Smooth NN or sparse/additive model; ensemble after validation"],
            ],
            widths=[1.7 * inch, 5.1 * inch],
        ),
        h2("Step 8: validate independently"),
        bullet("Use new random seeds and a stronger benchmark."),
        bullet("Report value, signed error, relative error, and benchmark standard error."),
        bullet("Slice results by maturity, payoff region, and state-space boundary."),
        PageBreak(),
    ]
)

story.extend(
    [
        h1("15A. Instrument-by-instrument proxy recipes"),
        p(
            "Each row below is a reproducible recipe. The state is what must be known at "
            "the valuation date; the label column describes the expensive price calculation; "
            "the transform and fitter define the cheap proxy."
        ),
        table(
            [
                ["Instrument", "State / coordinate", "Training label", "Target and fitter"],
                [
                    "European",
                    "Spot S; transform to d1",
                    "Shifted antithetic terminal GBM payoff",
                    "log(V+eps), PCHIP",
                ],
                [
                    "Asian",
                    "(S,A); reduce to adjusted moneyness",
                    "Path MC + geometric-Asian control",
                    "log(V/S) and log(time/S), PCHIP",
                ],
                [
                    "10-asset basket Asian",
                    "10 spots plus running basket sum",
                    "Sobol path MC; moment baseline",
                    "PCA sparse-Chebyshev + PCHIP residual calibration",
                ],
                [
                    "American",
                    "Spot S at each exercise date",
                    "One-step MC conditional continuation",
                    "Raw continuation PCHIP; max with intrinsic",
                ],
                [
                    "Barrier",
                    "Spot S plus alive/hit flag",
                    "Antithetic MC + bridge survival weight",
                    "log(V+eps), PCHIP",
                ],
            ],
            widths=[1.0 * inch, 1.75 * inch, 2.2 * inch, 1.85 * inch],
            font_size=6.9,
        ),
        h2("Recipe details: European"),
        bullet("Choose a valuation date and compute remaining maturity tau."),
        bullet("Generate spot nodes uniformly in d1 so both delta wings are covered."),
        bullet("At each node, importance-shift terminal normals toward the strike."),
        bullet("Fit log price with PCHIP; use the payoff exactly at maturity."),
        bullet("Validate against Black-Scholes on a denser shifted grid."),
        h2("Recipe details: arithmetic Asian"),
        bullet("Store spot S and running sum A before today's fixing."),
        bullet("Derive K_adj=(N K-A-S)/m and the scalar adjusted-moneyness coordinate."),
        bullet("Use antithetics, likelihood shifting, and the exact geometric Asian control."),
        bullet("Fit normalized value and time value; use the exact linear wing."),
        bullet("Validate with 500,000 independent paths per state."),
        h2("Recipe details: ten-asset basket Asian"),
        bullet("Store all ten spots and the running sum of previous equal-weight basket fixings."),
        bullet("Compute exact first and second moments of the final arithmetic basket average."),
        bullet("Use the moment-matched lognormal value as a baseline, not as the final proxy."),
        bullet("Add PCA scores of log spots to represent cross-sectional composition."),
        bullet("Fit bounded sparse-Chebyshev corrections, then calibrate residual bias with PCHIP."),
        bullet("Use a Gaussian likelihood-ratio mixture for long-dated OTM slices when it lowers weighted variance."),
        bullet("Fall back to plain Sobol on slices where the proposed shift increases variance."),
        PageBreak(),
        h1("15B. Exercise, barrier, and cliquet recipes"),
        h2("Recipe details: American put"),
        bullet("Start from the terminal payoff and move backward through exercise dates."),
        bullet("Simulate one-step transitions at 121 spot nodes."),
        bullet("Average the discounted next-date proxy to estimate continuation."),
        bullet("Fit continuation with PCHIP, then impose V=max(intrinsic, continuation)."),
        bullet("Validate against an independent projected finite-difference solution."),
        h2("Recipe details: barrier call"),
        bullet("Condition on whether the barrier has already been hit."),
        bullet("For discrete monitoring, check simulated prices only at monitoring dates."),
        bullet("For continuous monitoring, multiply exact segment bridge-survival weights."),
        bullet("Fit the alive-state log price in spot; obtain zero-rebate knock-in by parity."),
        h2("Recipe details: cliquet family"),
        table(
            [
                ["Instrument", "State", "Variance reduction", "Proxy"],
                [
                    "GBM cliquet",
                    "Accrued clipped return",
                    "Antithetic + clipped-sum control",
                    "Bounded logit Chebyshev d19",
                ],
                [
                    "Single SLV",
                    "Accrued, spot, variance",
                    "Antithetic + lower-tail shift",
                    "Local / anisotropic Chebyshev",
                ],
                [
                    "3-asset SLV",
                    "Accrued, 3 spots, 3 variances",
                    "Antithetic + mixture shift",
                    "Sparse bounded Chebyshev",
                ],
            ],
            widths=[1.3 * inch, 1.75 * inch, 2.0 * inch, 1.75 * inch],
            font_size=7.0,
        ),
        bullet("Compute exact global-floor and global-cap states before fitting."),
        bullet("Transform the remaining bounded price to logit space."),
        bullet("Concentrate states around floor/cap transition regions."),
        bullet("For the 7D basket, validate on multiple independent state-space designs."),
        p(
            "The 7D basket is the only family that did not reliably meet the 5-8% maximum-"
            "error goal. The documented default is the strongest fixed baseline, not a "
            "claim that the dimensionality problem has been solved."
        ),
        PageBreak(),
    ]
)

story.extend(
    [
        h1("16. Why there is no universal fitter"),
        p(
            "PCHIP works well for the American continuation curve because the geometry is "
            "one-dimensional, monotone, and kinked. It is not directly defined on a general "
            "multidimensional cloud. Tensor-product PCHIP would grow exponentially with "
            "dimension and requires a structured grid."
        ),
        p(
            "Chebyshev and Bernstein bases are excellent global smoothers in one dimension. "
            "In seven dimensions, unrestricted tensor bases are combinatorial. Sparse term "
            "selection helps, but sharp payoff transitions and interactions can still "
            "produce bias."
        ),
        p(
            "Neural networks are a genuinely generic high-dimensional candidate because "
            "parameter count can grow without a tensor grid. Their success requires enough "
            "state diversity, regularization, architecture tuning, and independent seeds. "
            "The small basket network tested here did not beat the structured ensemble."
        ),
        p(
            "A robust universal <i>workflow</i> is more realistic than a universal fitter: "
            "state reduction, payoff-aware features, bounded targets, variance-reduced "
            "labels, dimension-appropriate smoothers, and independent validation."
        ),
        h2("Recommended hierarchy"),
        table(
            [
                ["Question", "Action"],
                ["Can the state be proved 1D?", "Use and compare global ridge, Akima, PCHIP."],
                ["Are hard bounds known?", "Use exact tails and bounded-logit target."],
                ["Is there a dominant baseline?", "Fit residual or time value."],
                ["Do local/global models err oppositely?", "Use a simple cross-validated blend."],
                ["Does dimension exceed sparse-basis capacity?", "Increase state coverage and test smooth NN."],
            ],
            widths=[2.25 * inch, 4.55 * inch],
        ),
        PageBreak(),
    ]
)

# Error and implementation
story.extend(
    [
        h1("17. Error accounting"),
        h2("17.1 Metrics"),
        eq(
            """
signed_error = proxy - benchmark
absolute_error = |signed_error|
relative_error = absolute_error / max(|benchmark|, 0.01)
noise_ratio = absolute_error / max(benchmark_SE, tiny).
"""
        ),
        p(
            "A low noise ratio indicates the discrepancy may be statistically unresolved. "
            "A large noise ratio indicates model/fitting bias rather than benchmark noise."
        ),
        h2("17.2 Error decomposition"),
        eq(
            """
proxy - true continuous-model value
= fitting error
 + training MC error
 + benchmark estimation comparison noise
 + time-discretization bias
 + model/calibration error.
"""
        ),
        p(
            "The reported studies isolate fitting plus training noise against a benchmark "
            "under the same model discretization. They do not claim market calibration "
            "accuracy."
        ),
        h2("17.3 Leakage controls"),
        bullet("Independent training and benchmark random streams."),
        bullet("Fixed validation states not reused as training states."),
        bullet("A final standalone run with an independent seed where provided."),
        bullet("Method-selection results should be confirmed on a second validation design."),
        h2("17.4 Stability checks"),
        bullet("Repeat labels with different seeds."),
        bullet("Double Euler steps for SLV and compare."),
        bullet("Expand spot/variance domains to test extrapolation."),
        bullet("Check price bounds and monotonicity numerically."),
        bullet("For Greeks, validate derivative smoothness separately from price error."),
        PageBreak(),
    ]
)

story.extend(
    [
        h1("18. Reproducibility map"),
        table(
            [
                ["Purpose", "Entry point or folder"],
                ["European default", "EuroMain.py"],
                ["Asian default", "AsianMain.py"],
                ["10-asset basket Asian", "BasketAsianMain.py"],
                ["Barrier research/default", "BarrierOptExperiment/BarrierMain.py"],
                ["GBM cliquet default", "CliquetMain.py"],
                ["American put default", "AmericanMain.py"],
                ["Single-name SLV cliquet", "SLVCliquetMain.py"],
                ["3-asset SLV basket cliquet", "BasketCliquetOptExperiment/BasketCliquetMain.py"],
                ["Expanded 1D comparison", "OneDimensionalFitExperiment/ExpandedSplineStudy.py"],
                ["European research", "EuroOptExperiment/"],
                ["Asian research", "AsianOptExperiment/"],
                ["Cliquet research", "CliquetOptExperiment/"],
                ["American research", "AmericanOptExperiment/"],
                ["SLV research", "SLVCliquetOptExperiment/ and BasketCliquetOptExperiment/"],
            ],
            widths=[2.0 * inch, 4.8 * inch],
            font_size=7.3,
        ),
        h2("Core numerical budgets"),
        table(
            [
                ["Study", "Training", "Benchmark"],
                ["European", "121 states x 25,000 shifted paths", "Black-Scholes closed form"],
                ["Asian", "about 10M scenarios per fitted date", "500K paths/state + geometric control"],
                ["10-asset basket Asian", "513 states x 65,536 Sobol/LR paths/state", "524,288 Sobol/LR paths/state"],
                ["GBM cliquet", "about 10M scenarios per fitted date", "500K paths/state + clipped-sum control"],
                ["American", "10,006,700 one-step transitions", "4,000 x 2,000 projected FD grid"],
                ["Barrier", "10M scenarios per fitted date", "500K paths/state + bridge survival"],
                ["Single SLV cliquet", "about 10M scenarios per fitted date", "500K paths/state"],
                ["Basket SLV cliquet", "about 10M scenarios over 2,001 states/date", "500K paths x 31 states/date"],
            ],
            widths=[1.55 * inch, 2.65 * inch, 2.6 * inch],
            font_size=7.0,
        ),
        h2("Known limitations"),
        bullet("Illustrative SLV leverage functions are not calibrated market surfaces."),
        bullet("SLV uses two Euler steps per monthly period in the current experiments."),
        bullet("Basket Asian uses conditional LR mixture IS; it is deliberately off on dates where it raised variance."),
        bullet("Basket 5-8% control failed on untouched designs; adaptive enrichment remains research."),
        bullet("Pointwise price accuracy does not replace outer-scenario PFE quantile validation."),
        bullet("Production extrapolation, Greeks, and calibration loops require separate tests."),
        bullet("The neural basket test is not a definitive neural-network architecture search."),
        PageBreak(),
    ]
)

# Conclusion
story.extend(
    [
        Spacer(1, 0.4 * inch),
        h1("19. Final conclusions"),
        p(
            "The strongest improvement repeatedly came from changing the representation "
            "before changing the regressor. d1 organizes European wings; adjusted strike "
            "reduces the GBM Asian state; expected-total cushions organize cliquet bounds; "
            "the Snell recursion isolates American continuation."
        ),
        p(
            "The expanded one-dimensional test covered 99 European, American, Asian, and "
            "barrier cases with 17 estimators. Natural cubic interpolation led balanced "
            "p99 by only 0.035 percentage points, while PCHIP won 53 of 99 head-to-head "
            "cases and had zero local overshoot."
        ),
        p(
            "When one implementation must cover every genuinely one-dimensional option, "
            "PCHIP is therefore the universal default. Product-specific optimized profiles "
            "remain available when maximum pointwise accuracy matters. For PFE, the final "
            "acceptance metric must be exposure-quantile distortion in actual outer "
            "scenarios, not price RMSE alone."
        ),
        p(
            "For higher-dimensional SLV baskets, sparse payoff-aware regression was the "
            "strongest generic baseline, but three independent state designs showed that "
            "the 10M-path budget did not guarantee the 5-8% target. Robust worst errors "
            "were approximately 18.5%, 11.6%, and 9.9%."
        ),
        p(
            "For the ten-asset basket Asian, the successful generic pattern was different: "
            "use an analytically moment-matched low-fidelity baseline, summarize the "
            "remaining composition with PCA, fit a sparse Chebyshev log-factor correction, "
            "and then use PCHIP as a one-dimensional residual calibrator. Conditional "
            "LR mixture importance sampling improved the long-dated OTM slice. That "
            "fixed default achieved 5.873% worst max error against 524,288-path "
            "benchmarks."
        ),
        p(
            "Moment-residual Hermite, local, Nystrom Matern, weighted spectral, and fixed "
            "ensemble alternatives did not improve the 7D basket tail. This supports "
            "adaptive state enrichment or a larger path-level neural program rather than "
            "further tuning on the same sparse labels."
        ),
        p(
            "The reusable methodology is therefore: derive the state, reduce dimension when "
            "mathematically valid, enforce exact structure, create unbiased low-noise labels, "
            "fit with a dimension-appropriate smoother, and validate independently in the "
            "wings and transition regions.",
            "Callout",
        ),
        p(
            "The remaining pages are theorem statements, proofs, and references. They are "
            "not required to run the code, but they explain why each step is mathematically "
            "valid.",
            "Subtitle",
        ),
        PageBreak(),
    ]
)

# Theorem and proof appendices
story.extend(
    [
        h1("Appendix A. Probability and Monte Carlo theorems"),
        p(
            "The theorem statements below are restated in the notation of this project; "
            "they are not verbatim quotations. Each statement lists the assumptions used "
            "by the implementation."
        ),
        h2("A.1 Sample mean and standard error"),
        theorem(
            "Law of Large Numbers and Central Limit Theorem",
            "Let Y_1,Y_2,... be independent, identically distributed path values with "
            "finite mean mu and finite nonzero variance s^2. Then the sample mean converges "
            "to mu in probability. Moreover, sqrt(N)(Y_bar-mu)/s converges in distribution "
            "to a standard normal random variable.",
        ),
        proof(
            "The weak law follows from Var(Y_bar)=s^2/N and Chebyshev's inequality: "
            "P(|Y_bar-mu|>a) <= s^2/(N a^2), which tends to zero. The central-limit "
            "statement follows by expanding the characteristic function of the centered, "
            "standardized variable near zero and raising it to the Nth power. Replacing s "
            "by the sample standard deviation is valid because that estimator is consistent."
        ),
        h2("A.2 Conditioning and variance reduction"),
        theorem(
            "Tower Property and Rao-Blackwell Variance Identity",
            "For square-integrable Y and information G, E[E[Y|G]]=E[Y] and "
            "Var(Y)=E[Var(Y|G)]+Var(E[Y|G]). Therefore Var(E[Y|G])<=Var(Y).",
        ),
        proof(
            "The defining property of conditional expectation gives the first identity "
            "by choosing the whole sample space as the event. Write "
            "Y-E[Y]=(Y-E[Y|G])+(E[Y|G]-E[Y]). The cross term has expectation zero after "
            "conditioning on G, so expanding the square gives the variance identity. "
            "Brownian-bridge survival weighting uses exactly this idea."
        ),
        h2("A.3 Control variates"),
        theorem(
            "Optimal Linear Control",
            "Let Y be the discounted target payoff and C a control with known mean m_C and "
            "positive variance. The estimator Y_beta=Y-beta(C-m_C) is unbiased for every "
            "beta. Its variance is minimized by beta*=Cov(Y,C)/Var(C).",
        ),
        proof(
            "Unbiasedness follows because E[C-m_C]=0. Expanding the variance gives "
            "Var(Y_beta)=Var(Y)+beta^2 Var(C)-2 beta Cov(Y,C), a convex quadratic. "
            "Differentiating and setting the derivative to zero gives beta*."
        ),
        h2("A.4 Gaussian likelihood-ratio shift"),
        theorem(
            "Normal Mean Shift",
            "If Z is sampled from N(theta,1), then for any integrable payoff f, "
            "E_N(0,1)[f(Z)]=E_N(theta,1)[f(Z) exp(-theta Z+theta^2/2)].",
        ),
        proof(
            "Divide the standard-normal density phi(z) by the shifted density "
            "phi(z-theta). Completing the square gives their ratio "
            "exp(-theta z+theta^2/2). Multiplying the shifted integral by this ratio "
            "recovers the original integral exactly. The multistep formula multiplies "
            "these ratios, which adds their log likelihoods."
        ),
        h2("A.5 Antithetic unbiasedness"),
        theorem(
            "Antithetic Pairing",
            "If Z and -Z have the same distribution, then [f(Z)+f(-Z)]/2 has the same "
            "expectation as f(Z). Its variance is no greater than plain two-path averaging "
            "when Cov(f(Z),f(-Z))<=0.",
        ),
        proof(
            "Symmetry gives E[f(-Z)]=E[f(Z)]. The paired variance is one half of "
            "Var(f(Z)) plus one half of the covariance term, so a nonpositive covariance "
            "cannot increase variance."
        ),
        PageBreak(),
    ]
)

story.extend(
    [
        h1("Appendix B. Risk-neutral pricing and GBM"),
        h2("B.1 Risk-neutral pricing theorem"),
        theorem(
            "Risk-Neutral Valuation",
            "In an arbitrage-free complete market with money-market numeraire B_t, there "
            "is a unique probability measure Q under which every traded discounted price "
            "S_t/B_t is a martingale. A replicable payoff H at T has value "
            "V_t=B_t E_Q[H/B_T | F_t].",
        ),
        proof(
            "A self-financing replicating portfolio has discounted value equal to a "
            "Q-martingale, so its current value is the conditional expectation of its "
            "discounted terminal value. At T that terminal value equals H. If the option "
            "had a different price, buying the cheaper claim and selling the dearer "
            "replicating strategy would create an arbitrage. Existence and uniqueness of Q "
            "are the finite-market content of the fundamental theorem of asset pricing."
        ),
        h2("B.2 Ito's lemma"),
        theorem(
            "Ito's Lemma, One Dimension",
            "If X_t satisfies dX_t=a(t,X_t)dt+b(t,X_t)dW_t and f has one continuous time "
            "derivative and two continuous space derivatives, then "
            "df=(f_t+a f_x+0.5 b^2 f_xx)dt+b f_x dW_t.",
        ),
        proof(
            "Use a second-order Taylor expansion over a short interval. Brownian increments "
            "have size sqrt(dt), so (dW)^2 contributes at order dt, while dt*dW and dt^2 "
            "are smaller. The quadratic variation identity (dW)^2=dt produces the "
            "extra 0.5 b^2 f_xx term. A rigorous proof takes limits in probability over "
            "partitions; that measure-theory step is beyond the assumed prerequisites."
        ),
        h2("B.3 Exact GBM solution"),
        theorem(
            "Geometric Brownian Motion",
            "If dS/S=(r-q)dt+sigma dW with constant coefficients, then over tau, "
            "S_T=S_t exp[(r-q-sigma^2/2)tau+sigma sqrt(tau)Z], Z~N(0,1).",
        ),
        proof(
            "Apply Ito's lemma to f(S)=log S. Since f'=1/S and f''=-1/S^2, "
            "d log S=(r-q-sigma^2/2)dt+sigma dW. Integrate from t to T and use "
            "W_T-W_t=sqrt(tau)Z, then exponentiate."
        ),
        h2("B.4 Black-Scholes call formula"),
        theorem(
            "Black-Scholes-Merton Call",
            "Under the GBM assumptions above, a European call has value "
            "C=S exp(-q tau)Phi(d1)-K exp(-r tau)Phi(d2), where "
            "d1=[log(S/K)+(r-q+sigma^2/2)tau]/(sigma sqrt(tau)) and "
            "d2=d1-sigma sqrt(tau).",
        ),
        proof(
            "Risk-neutral pricing gives exp(-r tau)E[(S_T-K)+]. Split the expectation into "
            "E[S_T 1_(S_T>K)]-K P(S_T>K). The lognormal threshold is Z>-d2, giving "
            "P=Phi(d2). Completing the square inside the first normal integral shifts the "
            "threshold by sigma sqrt(tau), giving E[S_T 1]=S exp((r-q)tau)Phi(d1). "
            "Discounting produces the formula."
        ),
        h2("B.5 Correlated GBM second moment"),
        theorem(
            "Two-Asset GBM Moment",
            "Let S_i(t)=S_i(0) exp(mu_i t+sigma_i W_i(t)) with "
            "Corr(dW_i,dW_k)=rho_ik. Then "
            "E[S_i(t_a)S_k(t_b)]=S_i(0)S_k(0) exp(mu_i t_a+mu_k t_b+"
            "rho_ik sigma_i sigma_k min(t_a,t_b)).",
        ),
        proof(
            "The product equals S_i(0)S_k(0) times the exponential of a normal variable "
            "X=mu_i t_a+mu_k t_b+sigma_i W_i(t_a)+sigma_k W_k(t_b). Its variance is "
            "sigma_i^2 t_a+sigma_k^2 t_b+2 rho_ik sigma_i sigma_k min(t_a,t_b). "
            "For normal X, E[exp(X)]=exp(E[X]+0.5 Var(X)). In the risk-neutral GBM "
            "parameterization the log drift mu_l already includes -0.5 sigma_l^2, so "
            "the two single-asset 0.5 variance terms cancel those drift corrections, "
            "leaving exactly the cross term shown above."
        ),
        PageBreak(),
    ]
)

story.extend(
    [
        h1("Appendix C. Path-dependent identities and barriers"),
        h2("C.1 Asian homogeneity reduction"),
        theorem(
            "GBM Asian State Reduction",
            "At a fixed fixing date with m future fixings, define "
            "K_adj=(N K-A-S)/m. Under GBM, the arithmetic Asian call value satisfies "
            "V(S,A)=S times a function only of K_adj/S and time.",
        ),
        proof(
            "Future spots can be written S G_k, where the joint growth factors G_k do not "
            "depend on S. The payoff equals (m/N)[S average(G)-K_adj]+. Pulling S>0 "
            "outside the positive part gives S(m/N)[average(G)-K_adj/S]+. Taking the "
            "discounted expectation leaves a scalar function of K_adj/S."
        ),
        h2("C.2 Lognormal geometric Asian control"),
        theorem(
            "Weighted Normal Sum",
            "Any fixed linear combination of jointly normal variables is normal. Therefore "
            "the log of a discrete geometric average of GBM fixings is normal.",
        ),
        proof(
            "Each future log spot is an affine function of Gaussian increments. Their "
            "average is another affine combination of the same increments, hence normal "
            "by the defining closure property of multivariate normal vectors. Exponentiating "
            "makes the geometric average lognormal, so its option value is available in "
            "Black-Scholes form."
        ),
        h2("C.3 In/out parity"),
        theorem(
            "Zero-Rebate Barrier Parity",
            "For identical strike, maturity, barrier, and monitoring, a zero-rebate "
            "knock-in plus the corresponding knock-out equals the vanilla option path by "
            "path. Hence V_in+V_out=V_vanilla.",
        ),
        proof(
            "Every path is in exactly one of two disjoint events: the barrier is hit or it "
            "is not. On a hit path only the knock-in pays the vanilla payoff; on a no-hit "
            "path only the knock-out pays it. Add the two path payoffs and then take the "
            "discounted expectation."
        ),
        h2("C.4 Brownian-bridge survival"),
        theorem(
            "Single Lower-Barrier Bridge",
            "Let a Brownian bridge with variance rate sigma^2 start at x>b and end at y>b "
            "after dt. Its probability of staying above b is "
            "1-exp[-2(x-b)(y-b)/(sigma^2 dt)].",
        ),
        proof(
            "For Brownian motion starting at x, the reflection principle maps every path "
            "that first hits b and ends at y>b to a path ending at the reflected point "
            "2b-y. The ratio of the reflected Gaussian transition density to the direct "
            "density is exp[-2(x-b)(y-b)/(sigma^2 dt)]. Conditioning on the endpoint "
            "turns this ratio into the crossing probability; subtract from one."
        ),
        h2("C.5 Double-barrier image expansion"),
        theorem(
            "Absorbing Interval Kernel",
            "For Brownian motion killed at lower a and upper c, the transition density "
            "inside (a,c) equals an alternating infinite sum of free Gaussian densities "
            "at repeatedly reflected image points. Dividing this density by the free "
            "transition density gives conditional double-barrier survival.",
        ),
        proof(
            "Reflect the endpoint across a to enforce zero density at the lower boundary, "
            "then repeat reflections every 2(c-a) to enforce the upper boundary as well. "
            "At either boundary the direct and reflected terms cancel pairwise. Each term "
            "solves the heat equation, and the sum has the correct initial mass; uniqueness "
            "of the absorbing heat-equation solution identifies the kernel. The code "
            "truncates the rapidly decaying image sum and clips roundoff to [0,1]."
        ),
        PageBreak(),
    ]
)

story.extend(
    [
        h1("Appendix D. American exercise and exact payoff bounds"),
        h2("D.1 Bellman recursion"),
        theorem(
            "Finite-Horizon Optimal Stopping",
            "For exercise dates k=0,...,M and discounted reward g_k, define "
            "V_M=g_M and V_k=max(g_k, E[V_(k+1)|F_k]). Then V_k is the largest value "
            "obtainable by any stopping rule from date k, and the first date where "
            "g_k>=E[V_(k+1)|F_k] is optimal.",
        ),
        proof(
            "Use backward induction. At the last date, stopping is the only action. Assume "
            "V_(k+1) is optimal from the next date. At k there are only two choices: stop "
            "for g_k, or continue and receive conditional expected value E[V_(k+1)|F_k]. "
            "Taking their maximum is therefore optimal. Induction proves the result at all "
            "dates. This is the discrete Snell-envelope construction."
        ),
        h2("D.2 Cliquet floor and cap tails"),
        theorem(
            "Bound Propagation",
            "If m coupons remain and every coupon lies in [l,u], then final accumulated "
            "return lies in [a+m l,a+m u]. If this whole interval is below the global "
            "floor or above the global cap, the discounted payoff is known exactly.",
        ),
        proof(
            "Adding m quantities each between l and u adds between m l and m u. If the "
            "largest possible final return is below the floor, clipping always returns "
            "the floor. If the smallest possible final return is above the cap, clipping "
            "always returns the cap. No simulation is needed in either region."
        ),
        h2("D.3 Positive semidefinite factor construction"),
        theorem(
            "Factor-Generated Covariance",
            "If epsilon has identity covariance and Z=A epsilon, then Cov(Z)=A A' is "
            "positive semidefinite.",
        ),
        proof(
            "For every vector c, c'Cov(Z)c=c'A A'c=|A'c|^2>=0. The basket SLV shock "
            "construction is a linear factor model of this form, so it cannot create an "
            "invalid negative-variance direction."
        ),
        PageBreak(),
    ]
)

story.extend(
    [
        h1("Appendix E. Regression and sparse Chebyshev proofs"),
        h2("E.1 Ridge regression solution"),
        theorem(
            "Ridge Normal Equation",
            "For objective J(beta)=|A beta-y|^2+lambda|Gamma beta|^2, every minimizer "
            "satisfies (A'A+lambda Gamma'Gamma)beta=A'y. If that matrix is positive "
            "definite, the minimizer is unique.",
        ),
        proof(
            "Differentiate the quadratic: grad J=2A'(A beta-y)+2lambda Gamma'Gamma beta. "
            "Setting it to zero gives the equation. The Hessian is twice the coefficient "
            "matrix. Positive definiteness makes J strictly convex, so its stationary "
            "point is the unique global minimizer."
        ),
        h2("E.2 Chebyshev recurrence and boundedness"),
        theorem(
            "Chebyshev Cosine Identity",
            "For x=cos(theta), T_n(x)=cos(n theta). Consequently |T_n(x)|<=1 on [-1,1] "
            "and T_(n+1)=2x T_n-T_(n-1).",
        ),
        proof(
            "The cosine addition identity gives "
            "cos((n+1)theta)=2cos(theta)cos(n theta)-cos((n-1)theta), proving the "
            "recurrence. The bound follows because an ordinary cosine lies in [-1,1]. "
            "This bounded basis is numerically safer than powers x^n on a wide raw domain."
        ),
        h2("E.3 Curse of the full tensor"),
        theorem(
            "Tensor Term Count",
            "If each of D coordinates may use degrees 0 through p independently, the full "
            "tensor polynomial basis contains (p+1)^D terms.",
        ),
        proof(
            "There are p+1 choices for each coordinate degree alpha_j. The multiplication "
            "principle gives (p+1) multiplied by itself D times. For D=7 and p=5 this is "
            "6^7=279,936 terms before any regression is solved."
        ),
        h2("E.4 Sparse anisotropic index sets"),
        theorem(
            "Downward-Closed Sparse Set",
            "If I={alpha: Sum_j w_j alpha_j<=p} with positive weights, then alpha in I and "
            "0<=beta_j<=alpha_j imply beta in I.",
        ),
        proof(
            "Because all weights are positive, Sum_j w_j beta_j<=Sum_j w_j alpha_j<=p. "
            "Thus lower-order parent terms are retained whenever a higher-order term is "
            "retained, which makes the sparse hierarchy interpretable."
        ),
        PageBreak(),
    ]
)

story.extend(
    [
        h1("Appendix F. PCHIP and cubic interpolation theorems"),
        h2("F.1 Unique cubic Hermite segment"),
        theorem(
            "Hermite Interpolation",
            "Given two distinct endpoints, two endpoint values, and two endpoint slopes, "
            "there is exactly one polynomial of degree at most three satisfying all four "
            "conditions.",
        ),
        proof(
            "The four Hermite basis functions displayed in Section 6 construct one such "
            "polynomial. If two cubics satisfied the conditions, their difference would "
            "have a double root at each endpoint, hence at least four roots counting "
            "multiplicity. A nonzero cubic cannot have four roots, so the difference is zero."
        ),
        h2("F.2 Local PCHIP monotonicity"),
        theorem(
            "Fritsch-Carlson / Fritsch-Butland Monotone Cubic",
            "For strictly ordered nodes with monotone data, choosing interior derivatives "
            "by the sign test and weighted harmonic mean, together with the stated endpoint "
            "limiters, produces a monotone piecewise cubic Hermite interpolant.",
        ),
        proof(
            "On one interval, subtract y_i and divide by the nonzero secant so the endpoint "
            "values become 0 and 1. The derivative of the Hermite cubic is then a quadratic "
            "whose coefficients depend only on the two normalized endpoint slopes. "
            "Substituting the harmonic-mean slopes and endpoint bounds places those "
            "normalized slopes in the monotonicity region derived by Fritsch and Carlson, "
            "so the quadratic is nonnegative on [0,1] for increasing data and nonpositive "
            "for decreasing data. If adjacent secants change sign, the zero derivative "
            "joins the two monotone pieces without an extra extremum. The original paper "
            "contains the complete case-by-case quadratic inequalities."
        ),
        h2("F.3 Bezier and Hermite equivalence"),
        theorem(
            "Cubic Representation Equivalence",
            "A cubic Bezier curve with control points P0=y_i, P1=y_i+h d_i/3, "
            "P2=y_(i+1)-h d_(i+1)/3, P3=y_(i+1) is exactly the cubic Hermite segment "
            "with endpoint values y_i,y_(i+1) and slopes d_i,d_(i+1).",
        ),
        proof(
            "Expand the four Bernstein polynomials in powers of t and collect the "
            "coefficients of y_i, h d_i, y_(i+1), and h d_(i+1). They are respectively "
            "2t^3-3t^2+1, t^3-2t^2+t, -2t^3+3t^2, and t^3-t^2, exactly the Hermite basis."
        ),
        PageBreak(),
    ]
)

story.extend(
    [
        h1("Appendix G. Bounded transforms and moment formulas"),
        h2("G.1 Logit enforces price bounds"),
        theorem(
            "Logistic Bijection",
            "The map logistic(y)=1/(1+exp(-y)) sends every real y into (0,1), is strictly "
            "increasing, and has inverse log(p/(1-p)). Therefore "
            "L+(U-L)logistic(y) always lies strictly between L and U.",
        ),
        proof(
            "The exponential is positive, so the denominator exceeds one and the fraction "
            "lies in (0,1). Differentiation gives logistic(y)[1-logistic(y)]>0. Solving "
            "p=1/(1+exp(-y)) for y gives the stated inverse. Affine rescaling maps (0,1) "
            "to (L,U)."
        ),
        h2("G.2 Expected clipped normal value"),
        theorem(
            "Clipped Normal Mean",
            "If X~N(mu,s^2), then E[clip(X,L,U)]=L+C(L)-C(U), where "
            "C(K)=(mu-K)Phi((mu-K)/s)+s phi((mu-K)/s).",
        ),
        proof(
            "Pathwise, clip(X,L,U)=L+(X-L)+-(X-U)+. For a normal variable, substitute "
            "X=mu+sZ in E[(X-K)+] and integrate over Z>(K-mu)/s. Splitting the integral "
            "into (mu-K) times a tail probability plus s times the first normal tail "
            "moment gives C(K)."
        ),
        h2("G.3 Common random numbers"),
        theorem(
            "Unbiased Common-Random-Number Labels",
            "Using the same random draws to estimate prices at several states does not "
            "change the expectation of any individual state estimator. It changes only "
            "their cross-state covariance.",
        ),
        proof(
            "At each fixed state, the reused random vector has exactly the same marginal "
            "distribution as a freshly drawn vector, so its sample mean remains unbiased. "
            "Sharing draws correlates errors across states; for nearby smooth payoffs this "
            "usually makes the error itself vary smoothly, which helps interpolation."
        ),
        PageBreak(),
    ]
)

story.extend(
    [
        h1("Appendix H. Mathematical implementation tricks"),
        p(
            "This appendix turns the main implementation choices into mathematical "
            "claims. The statements are intentionally compact; each proof explains why "
            "the code transformation is valid, unbiased, stabilizing, or only a heuristic."
        ),
        h2("H.1 State enrichment is not importance sampling"),
        theorem(
            "Two Different Measures",
            "Let x be the pricing state and Z the simulated path randomness. Choosing "
            "more training states from a tail-focused state distribution nu_train changes "
            "where labels are requested. It does not change the conditional expectation "
            "inside any label. Importance sampling changes the simulation law for Z and "
            "must include a density-ratio weight.",
        ),
        eq(
            """
State enrichment:
    x_i ~ nu_train,
    y_i = (1/N) sum_{n=1}^N h(x_i,Z_n),        Z_n ~ p_x.

Likelihood-ratio importance sampling:
    y_i = (1/N) sum_{n=1}^N h(x_i,Z_n) p_x(Z_n)/q_x(Z_n),
    Z_n ~ q_x.
"""
        ),
        proof(
            "For a fixed state x, the target label is E_p[h(x,Z)]. Drawing more states "
            "near a boundary only changes the empirical design measure used by the "
            "regression. It leaves each label estimator as an average under p_x. If paths "
            "are instead drawn from q_x, the expectation becomes E_q[h(x,Z)] unless the "
            "likelihood ratio p_x/q_x is multiplied pathwise. Therefore state enrichment "
            "improves coverage, while importance sampling improves rare-event label "
            "variance; they are related but not the same operation."
        ),
        h2("H.2 Sobol power-of-two batching"),
        theorem(
            "Base-2 Digital-Net Balance",
            "For a Sobol sequence in base 2, the first 2^m points form a digital net. "
            "Elementary binary boxes of compatible volume receive nearly uniform counts. "
            "Using powers of two therefore preserves the designed low-discrepancy balance "
            "better than stopping at an arbitrary path count.",
        ),
        eq(
            """
Implementation rule:
    N_target -> N_sobol = 2^{ceil(log_2 N_target)}.

Training budget:
    N_state * N_sobol paths.
"""
        ),
        proof(
            "Sobol points are constructed so that the first 2^m points fill binary "
            "elementary intervals evenly according to the generator matrices. Truncating "
            "at a non-power-of-two count cuts through this balanced block. Rounding up "
            "keeps the full net and only increases work by less than a factor of two. "
            "This is a numerical-design theorem, not a probabilistic unbiasedness claim."
        ),
        h2("H.3 PCA compression of basket composition"),
        theorem(
            "Best Linear k-Dimensional Reconstruction",
            "Let R be the centered log-spot vector with covariance matrix Sigma. Among "
            "all rank-k orthogonal projections P, the projection onto the eigenvectors "
            "of Sigma with the k largest eigenvalues minimizes E[||R-PR||^2].",
        ),
        eq(
            """
Sigma u_j = lambda_j u_j,    lambda_1 >= ... >= lambda_d.

P_k = U_k U_k',
score_j = R' u_j,       j=1,...,k.
"""
        ),
        proof(
            "For any orthogonal projection P with rank k, the retained variance is "
            "trace(P Sigma). Minimizing reconstruction error is equivalent to maximizing "
            "this retained variance because E[||R-PR||^2]=trace(Sigma)-trace(P Sigma). "
            "Rayleigh-Ritz gives the maximum trace as lambda_1+...+lambda_k, achieved by "
            "the span of the top eigenvectors. The basket Asian proxy uses the first four "
            "scores as a low-dimensional summary of relative basket composition."
        ),
        h2("H.4 Moment-matched lognormal baseline"),
        theorem(
            "Two-Moment Lognormal Match",
            "If a positive random variable A has mean m>0 and variance v>=0, the "
            "lognormal random variable exp(Y), Y~N(mu_L,s_L^2), with the same first two "
            "moments is obtained from s_L^2=log(1+v/m^2) and "
            "mu_L=log(m)-0.5 s_L^2.",
        ),
        eq(
            """
s_L^2 = log(1 + v/m^2),
mu_L = log(m) - s_L^2/2.

V_LN = exp(-r tau) [ m Phi(d_1) - K Phi(d_2) ],
d_2 = (mu_L - log K)/s_L,
d_1 = d_2 + s_L.
"""
        ),
        proof(
            "For Y~N(mu_L,s_L^2), E[exp(Y)]=exp(mu_L+s_L^2/2) and "
            "Var(exp(Y))=exp(2 mu_L+s_L^2)(exp(s_L^2)-1). Setting the mean equal to m "
            "gives exp(mu_L+s_L^2/2)=m. Dividing the variance equation by m^2 gives "
            "v/m^2=exp(s_L^2)-1, hence the stated s_L^2 and mu_L. The option formula "
            "then follows by the same truncated-lognormal calculation used in "
            "Black-Scholes."
        ),
        h2("H.5 Ridge residual fitting"),
        theorem(
            "Ridge Normal Equations",
            "Given design matrix X, target y, and positive penalty lambda, the "
            "coefficient vector minimizing ||X beta-y||_2^2+lambda||beta||_2^2 is "
            "beta=(X'X+lambda I)^{-1}X'y. If the intercept is unpenalized, set the first "
            "diagonal penalty entry to zero.",
        ),
        eq(
            """
min_beta  ||X beta - y||_2^2 + beta' Lambda beta

Gradient:
    2 X'(X beta-y) + 2 Lambda beta = 0

Solution:
    beta = (X'X + Lambda)^{-1} X'y.
"""
        ),
        proof(
            "The objective is a convex quadratic. Differentiating with respect to beta "
            "gives the displayed gradient. Setting it to zero yields the linear system. "
            "Adding lambda I makes the system better conditioned and shrinks unstable "
            "high-order basis coefficients. The code uses this for sparse Chebyshev and "
            "residual corrections."
        ),
        h2("H.6 Bounded log-factor correction"),
        theorem(
            "Positive Multiplicative Correction",
            "Let B(x)>=0 be a baseline price and eps>0. If a model fits "
            "a(x)=log((V(x)+eps)/(B(x)+eps)), then the reconstructed price "
            "V_hat(x)=(B(x)+eps) exp(a_hat(x))-eps is bounded below by -eps. Clipping at "
            "zero gives a nonnegative price.",
        ),
        eq(
            """
a(x) = log( (V(x)+eps)/(B(x)+eps) )

V_hat(x) = max( (B(x)+eps) exp(a_hat(x)) - eps, 0 ).
"""
        ),
        proof(
            "The exponential is strictly positive, so (B+eps) exp(a_hat) is positive. "
            "Subtracting eps can only make the raw reconstruction larger than -eps. "
            "The final maximum with zero enforces the no-negative-option-price condition. "
            "Fitting a ratio rather than raw value lets the high-dimensional model learn "
            "relative bias around a strong moment baseline."
        ),
        h2("H.7 PCHIP residual calibration"),
        theorem(
            "Conditional-Mean Residual Correction",
            "Let M be a scalar summary feature and R=(V-g(X))/D(X) be a normalized "
            "training residual for a preliminary proxy g and positive scale D. Among all "
            "square-integrable functions c(M), the function c*(M)=E[R|M] minimizes "
            "E[(R-c(M))^2].",
        ),
        eq(
            """
Preliminary proxy:     g(X)
Normalized residual:   R = (V - g(X)) / D(X)
Calibration function:  c*(m) = E[R | M=m]

Corrected proxy:
    V_cal(X) = max( g(X) + D(X) c_hat(M(X)), 0 ).
"""
        ),
        proof(
            "For any function c(M), condition on M. The conditional mean-square error is "
            "E[(R-c(M))^2 | M]. For a fixed value M=m this is minimized by the scalar "
            "mean E[R|M=m], because E[(R-a)^2|M=m]=Var(R|M=m)+(E[R|M=m]-a)^2. The code "
            "estimates c* by binning residuals in expected-average moneyness and fitting "
            "a PCHIP curve through the bin means. This is why the correction is a "
            "mathematical residual projection, not a hand-drawn patch."
        ),
        h2("H.8 Relative-error floor"),
        theorem(
            "Stable Tail Error Denominator",
            "For floor eta greater than zero, define e_rel=|V_hat-V|/max(|V|,eta). "
            "If |V| is at least eta, this is ordinary relative error. If |V| is below "
            "eta, then e_rel is bounded by |V_hat-V|/eta, so tiny "
            "option prices cannot create unbounded percentages from economically tiny "
            "absolute errors.",
        ),
        proof(
            "The first case follows from max(|V|,eta)=|V|. In the second case the "
            "denominator is eta, a fixed positive number. Thus the metric remains "
            "sensitive to absolute tail misses while avoiding division by a number "
            "arbitrarily close to zero."
        ),
        h2("H.9 Exact linear payoff wings"),
        theorem(
            "Positive-Part Removal",
            "If a call payoff can be written (Y-K)_+ and the state implies Y is at least K "
            "almost surely, then E[(Y-K)_+]=E[Y]-K. If the state implies Y is at most K, "
            "then E[(Y-K)_+]=0.",
        ),
        proof(
            "On the event Y is at least K, the positive part equals Y-K path by path. On "
            "the event Y is at most K, it equals zero path by path. Taking expectations preserves these "
            "identities. The scripts use this to avoid asking regressions to learn "
            "regions where the price is exactly linear or exactly zero."
        ),
        h2("H.10 Sparse Chebyshev basis selection"),
        theorem(
            "Bounded Basis on a Scaled Domain",
            "Chebyshev polynomials satisfy T_n(cos theta)=cos(n theta). Therefore "
            "|T_n(z)| is at most one for z in [-1,1]. Scaling inputs to this interval keeps each "
            "univariate basis column bounded before interactions are formed.",
        ),
        eq(
            """
Scale feature:
    z_j = 2 (x_j - low_j)/(high_j-low_j) - 1.

Sparse design:
    1, T_1(z_j),...,T_p(z_j), selected z_i z_j, selected z_i z_j^2.
"""
        ),
        proof(
            "Every z in [-1,1] can be written z=cos theta for some theta in [0,pi]. "
            "Then T_n(z)=cos(n theta), whose absolute value is at most one. The code uses "
            "quantile scaling and clipping, so most training and validation features lie "
            "inside or near this bounded interval. Sparse term selection is a bias-"
            "variance tradeoff: it gives up full tensor expressiveness to avoid the "
            "combinatorial explosion of all interactions."
        ),
        PageBreak(),
    ]
)

story.extend(
    [
        h1("Appendix I. Primary references"),
        p(
            "These sources support the named theorems and numerical methods. The appendix "
            "statements are project-specific restatements, not copied quotations."
        ),
        bullet(
            "Black, F. and Scholes, M. (1973), The Pricing of Options and Corporate "
            "Liabilities, Journal of Political Economy 81, 637-654. "
            "https://doi.org/10.1086/260062"
        ),
        bullet(
            "Fritsch, F. N. and Carlson, R. E. (1980), Monotone Piecewise Cubic "
            "Interpolation, SIAM Journal on Numerical Analysis 17, 238-246. "
            "https://doi.org/10.1137/0717021"
        ),
        bullet(
            "Fritsch, F. N. and Butland, J. (1984), A Method for Constructing Local "
            "Monotone Piecewise Cubic Interpolants, SIAM Journal on Scientific and "
            "Statistical Computing 5, 300-304."
        ),
        bullet(
            "Hoerl, A. E. and Kennard, R. W. (1970), Ridge Regression: Biased Estimation "
            "for Nonorthogonal Problems, Technometrics 12, 55-67. "
            "https://doi.org/10.1080/00401706.1970.10488634"
        ),
        bullet(
            "Eilers, P. H. C. and Marx, B. D. (1996), Flexible Smoothing with B-splines "
            "and Penalties, Statistical Science 11, 89-121. "
            "https://doi.org/10.1214/ss/1038425655"
        ),
        bullet(
            "Broadie, M., Glasserman, P., and Kou, S. G. (1997), A Continuity Correction "
            "for Discrete Barrier Options, Mathematical Finance 7, 325-348."
        ),
        bullet(
            "Glasserman, P. and Staum, J. (2001), Conditioning on One-Step Survival for "
            "Barrier Option Simulations, Operations Research 49, 923-937. "
            "https://doi.org/10.1287/opre.49.6.923.10018"
        ),
        bullet(
            "Reinsch, C. H. (1967), Smoothing by Spline Functions, Numerische Mathematik "
            "10, 177-183. https://doi.org/10.1007/BF02162161"
        ),
        bullet(
            "Floater, M. S. and Hormann, K. (2007), Barycentric Rational Interpolation "
            "with No Poles and High Rates of Approximation, Numerische Mathematik 107, "
            "315-331. https://doi.org/10.1007/s00211-007-0093-y"
        ),
        p(
            "Standard probability results (law of large numbers, central limit theorem, "
            "conditional expectation, and normal-density change of measure) are proved "
            "directly above in the finite-variance setting used by the simulations."
        ),
        PageBreak(),
    ]
)

def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.68 * inch,
        title="Monte Carlo Proxy Pricing Methodology",
        author="Proxy Pricing Research Project",
        subject="European, American, Asian, barrier, cliquet, and basket cliquet proxies",
    )
    document.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()
