"""
ml_sentences.py — Multilingual F1 sentences for semantic convergence experiment.

Provides accurate translations of the F1 sentences (2 per factor) for all 24 factors
into German (de), Spanish (es), Arabic MSA (ar), and Chinese Simplified (zh).

24 factors × 4 languages × 2 sentences = 192 sentences total.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# ML_SENTENCES: factor_id → lang → [sent1, sent2]
# ---------------------------------------------------------------------------
# Translations target the same propositional meaning as the English F1 sentences.
# Arabic: Modern Standard Arabic (MSA). Chinese: Simplified characters.
# ---------------------------------------------------------------------------

ML_SENTENCES = {

    # ── DOMAIN A ── Scalar temperature ──────────────────────────────────────

    "A1": {
        "de": [
            "Das Objekt ist heiß.",
            "Das Ding ist heiß.",
        ],
        "es": [
            "El objeto está caliente.",
            "La cosa está caliente.",
        ],
        "ar": [
            "الجسم حار.",
            "الشيء حار.",
        ],
        "zh": [
            "这个物体是热的。",
            "这个东西是热的。",
        ],
    },

    "A2": {
        "de": [
            "Das Objekt ist kalt.",
            "Das Ding ist kalt.",
        ],
        "es": [
            "El objeto está frío.",
            "La cosa está fría.",
        ],
        "ar": [
            "الجسم بارد.",
            "الشيء بارد.",
        ],
        "zh": [
            "这个物体是冷的。",
            "这个东西是冷的。",
        ],
    },

    "A3": {
        "de": [
            "Das Objekt ist warm.",
            "Das Ding ist warm.",
        ],
        "es": [
            "El objeto está tibio.",
            "La cosa está tibia.",
        ],
        "ar": [
            "الجسم دافئ.",
            "الشيء دافئ.",
        ],
        "zh": [
            "这个物体是温热的。",
            "这个东西是温热的。",
        ],
    },

    "A4": {
        "de": [
            "Das Objekt ist kühl.",
            "Das Ding ist kühl.",
        ],
        "es": [
            "El objeto está fresco.",
            "La cosa está fresca.",
        ],
        "ar": [
            "الجسم فاتر.",
            "الشيء فاتر.",
        ],
        "zh": [
            "这个物体是凉的。",
            "这个东西是凉的。",
        ],
    },

    # ── DOMAIN B ── Scalar size ──────────────────────────────────────────────

    "B1": {
        "de": [
            "Das Objekt ist groß.",
            "Das Ding ist groß.",
        ],
        "es": [
            "El objeto es grande.",
            "La cosa es grande.",
        ],
        "ar": [
            "الجسم كبير.",
            "الشيء كبير.",
        ],
        "zh": [
            "这个物体是大的。",
            "这个东西是大的。",
        ],
    },

    "B2": {
        "de": [
            "Das Objekt ist klein.",
            "Das Ding ist klein.",
        ],
        "es": [
            "El objeto es pequeño.",
            "La cosa es pequeña.",
        ],
        "ar": [
            "الجسم صغير.",
            "الشيء صغير.",
        ],
        "zh": [
            "这个物体是小的。",
            "这个东西是小的。",
        ],
    },

    "B3": {
        "de": [
            "Das Objekt ist riesig.",
            "Das Ding ist riesig.",
        ],
        "es": [
            "El objeto es enorme.",
            "La cosa es enorme.",
        ],
        "ar": [
            "الجسم ضخم جداً.",
            "الشيء ضخم جداً.",
        ],
        "zh": [
            "这个物体是巨大的。",
            "这个东西是巨大的。",
        ],
    },

    "B4": {
        "de": [
            "Das Objekt ist winzig.",
            "Das Ding ist winzig.",
        ],
        "es": [
            "El objeto es diminuto.",
            "La cosa es diminuta.",
        ],
        "ar": [
            "الجسم ضئيل جداً.",
            "الشيء ضئيل جداً.",
        ],
        "zh": [
            "这个物体是微小的。",
            "这个东西是微小的。",
        ],
    },

    # ── DOMAIN C ── Taxonomic / refinement ──────────────────────────────────

    "C1": {
        "de": [
            "Das Tier ist ein Hund.",
            "Das Lebewesen ist ein Hund.",
        ],
        "es": [
            "El animal es un perro.",
            "La criatura es un perro.",
        ],
        "ar": [
            "الحيوان كلب.",
            "المخلوق كلب.",
        ],
        "zh": [
            "这只动物是一只狗。",
            "这个生物是一只狗。",
        ],
    },

    "C2": {
        "de": [
            "Das Tier ist ein Canide.",
            "Das Lebewesen ist ein Canide.",
        ],
        "es": [
            "El animal es un cánido.",
            "La criatura es un cánido.",
        ],
        "ar": [
            "الحيوان من فصيلة الكلبيات.",
            "المخلوق من فصيلة الكلبيات.",
        ],
        "zh": [
            "这只动物是犬科动物。",
            "这个生物是犬科动物。",
        ],
    },

    "C3": {
        "de": [
            "Das Tier ist ein Säugetier.",
            "Das Lebewesen ist ein Säugetier.",
        ],
        "es": [
            "El animal es un mamífero.",
            "La criatura es un mamífero.",
        ],
        "ar": [
            "الحيوان من الثدييات.",
            "المخلوق من الثدييات.",
        ],
        "zh": [
            "这只动物是哺乳动物。",
            "这个生物是哺乳动物。",
        ],
    },

    "C4": {
        "de": [
            "Der Hund ist ein Arbeitshund.",
            "Dieser Hund ist ein Arbeitshund.",
        ],
        "es": [
            "El perro es un perro de trabajo.",
            "Este perro es un perro de trabajo.",
        ],
        "ar": [
            "الكلب كلب عمل.",
            "هذا الكلب كلب عمل.",
        ],
        "zh": [
            "这只狗是一只工作犬。",
            "这只狗是工作犬。",
        ],
    },

    # ── DOMAIN D ── Structured object / function ────────────────────────────

    "D1": {
        "de": [
            "Das Fahrzeug ist ein elektrisches Personenkraftfahrzeug.",
            "Dieses Fahrzeug ist ein elektrisches Auto für Passagiere.",
        ],
        "es": [
            "El vehículo es un vehículo de pasajeros eléctrico de carretera.",
            "Este vehículo es un automóvil eléctrico para pasajeros.",
        ],
        "ar": [
            "المركبة مركبة ركاب كهربائية على الطريق.",
            "هذه المركبة سيارة كهربائية لنقل الركاب.",
        ],
        "zh": [
            "这辆车是一辆电动载客公路车辆。",
            "这辆车是一辆供乘客乘坐的电动汽车。",
        ],
    },

    "D2": {
        "de": [
            "Das Fahrzeug ist ein Güterfahrzeug.",
            "Dieses Fahrzeug transportiert Güter.",
        ],
        "es": [
            "El vehículo es un vehículo de carga.",
            "Este vehículo transporta carga.",
        ],
        "ar": [
            "المركبة مركبة شحن.",
            "هذه المركبة تنقل البضائع.",
        ],
        "zh": [
            "这辆车是一辆货运车辆。",
            "这辆车运载货物。",
        ],
    },

    "D3": {
        "de": [
            "Das Fahrzeug ist ein Kompaktwagen.",
            "Dieses Auto ist ein Kompaktwagen.",
        ],
        "es": [
            "El vehículo es un automóvil compacto.",
            "Este automóvil es un automóvil compacto.",
        ],
        "ar": [
            "المركبة سيارة مدمجة.",
            "هذه السيارة سيارة مدمجة.",
        ],
        "zh": [
            "这辆车是一辆紧凑型轿车。",
            "这辆汽车是一辆紧凑型轿车。",
        ],
    },

    "D4": {
        "de": [
            "Das Softwaresystem ist eine verteilte Datenbank.",
            "Dieses System ist eine verteilte Datenbank.",
        ],
        "es": [
            "El sistema de software es una base de datos distribuida.",
            "Este sistema es una base de datos distribuida.",
        ],
        "ar": [
            "نظام البرمجيات قاعدة بيانات موزعة.",
            "هذا النظام قاعدة بيانات موزعة.",
        ],
        "zh": [
            "该软件系统是一个分布式数据库。",
            "该系统是一个分布式数据库。",
        ],
    },

    # ── DOMAIN E ── Human-designed object refinements ───────────────────────

    "E1": {
        "de": [
            "Das Schuhwerk ist ein wasserdichter Wanderstiefel.",
            "Dieser Schuh ist ein wasserdichter Wanderstiefel.",
        ],
        "es": [
            "El calzado es una bota de senderismo impermeable.",
            "Este zapato es una bota de senderismo impermeable.",
        ],
        "ar": [
            "الحذاء حذاء مشي مقاوم للماء.",
            "هذا الحذاء حذاء مشي جبلي مقاوم للماء.",
        ],
        "zh": [
            "这双鞋是一双防水登山靴。",
            "这只鞋是一双防水登山靴。",
        ],
    },

    "E2": {
        "de": [
            "Der Stuhl ist ein ergonomischer Bürostuhl.",
            "Dieser Sitz ist ein ergonomischer Bürostuhl.",
        ],
        "es": [
            "La silla es una silla de oficina ergonómica.",
            "Este asiento es una silla de oficina ergonómica.",
        ],
        "ar": [
            "الكرسي كرسي مكتبي مريح للجسم.",
            "هذا المقعد كرسي مكتبي مريح للجسم.",
        ],
        "zh": [
            "这把椅子是一把符合人体工程学的办公椅。",
            "这个座位是一把符合人体工程学的办公椅。",
        ],
    },

    "E3": {
        "de": [
            "Das Gebäude ist ein Wohnhochhaus.",
            "Dieses Bauwerk ist ein Wohnhochhaus.",
        ],
        "es": [
            "El edificio es un edificio de apartamentos residenciales.",
            "Esta estructura es un edificio de apartamentos residenciales.",
        ],
        "ar": [
            "المبنى مبنى شقق سكنية.",
            "هذا المبنى مبنى شقق سكنية.",
        ],
        "zh": [
            "这栋建筑是一栋住宅公寓楼。",
            "这个建筑是一栋住宅公寓楼。",
        ],
    },

    "E4": {
        "de": [
            "Das Gebäude ist ein Hochhaus mit Wohnungen.",
            "Dieses Bauwerk ist ein Hochhaus mit Wohnungen.",
        ],
        "es": [
            "El edificio es un edificio de apartamentos de gran altura.",
            "Esta estructura es un edificio de apartamentos de gran altura.",
        ],
        "ar": [
            "المبنى برج سكني شاهق.",
            "هذا المبنى برج سكني شاهق.",
        ],
        "zh": [
            "这栋建筑是一栋高层公寓楼。",
            "这个建筑是一栋高层公寓楼。",
        ],
    },

    # ── DOMAIN F ── Institutional / legal ───────────────────────────────────

    "F1": {
        "de": [
            "Die Institution ist eine öffentliche Forschungsuniversität.",
            "Diese Organisation ist eine öffentliche Forschungsuniversität.",
        ],
        "es": [
            "La institución es una universidad pública de investigación.",
            "Esta organización es una universidad pública de investigación.",
        ],
        "ar": [
            "المؤسسة جامعة بحثية حكومية.",
            "هذه المنظمة جامعة بحثية حكومية.",
        ],
        "zh": [
            "该机构是一所公立研究型大学。",
            "该组织是一所公立研究型大学。",
        ],
    },

    "F2": {
        "de": [
            "Die Institution ist eine öffentliche Universität.",
            "Diese Organisation ist eine öffentliche Universität.",
        ],
        "es": [
            "La institución es una universidad pública.",
            "Esta organización es una universidad pública.",
        ],
        "ar": [
            "المؤسسة جامعة حكومية.",
            "هذه المنظمة جامعة حكومية.",
        ],
        "zh": [
            "该机构是一所公立大学。",
            "该组织是一所公立大学。",
        ],
    },

    "F3": {
        "de": [
            "Das Dokument ist ein befristeter Arbeitsvertrag.",
            "Diese Vereinbarung ist ein befristeter Arbeitsvertrag.",
        ],
        "es": [
            "El documento es un contrato de trabajo de duración determinada.",
            "Este acuerdo es un contrato de trabajo de duración determinada.",
        ],
        "ar": [
            "الوثيقة عقد عمل محدد المدة.",
            "هذه الاتفاقية عقد عمل محدد المدة.",
        ],
        "zh": [
            "该文件是一份固定期限劳动合同。",
            "该协议是一份固定期限劳动合同。",
        ],
    },

    "F4": {
        "de": [
            "Das Dokument ist ein Arbeitsvertrag.",
            "Diese Vereinbarung ist ein Arbeitsvertrag.",
        ],
        "es": [
            "El documento es un contrato de trabajo.",
            "Este acuerdo es un contrato de trabajo.",
        ],
        "ar": [
            "الوثيقة عقد عمل.",
            "هذه الاتفاقية عقد عمل.",
        ],
        "zh": [
            "该文件是一份劳动合同。",
            "该协议是一份劳动合同。",
        ],
    },
}


# ---------------------------------------------------------------------------
# Factor metadata needed to build the dataframe
# ---------------------------------------------------------------------------

_FACTOR_META = {
    "A1": "A", "A2": "A", "A3": "A", "A4": "A",
    "B1": "B", "B2": "B", "B3": "B", "B4": "B",
    "C1": "C", "C2": "C", "C3": "C", "C4": "C",
    "D1": "D", "D2": "D", "D3": "D", "D4": "D",
    "E1": "E", "E2": "E", "E3": "E", "E4": "E",
    "F1": "F", "F2": "F", "F3": "F", "F4": "F",
}


def build_ml_dataframe() -> pd.DataFrame:
    """Return a DataFrame of all multilingual F1 sentences.

    Columns match the output of dataset.build_dataframe() plus a 'language' column.
    split = 'ml' so these are excluded from meaning-classifier train/val/test.
    """
    rows = []
    for factor_id, lang_dict in ML_SENTENCES.items():
        domain_id = _FACTOR_META[factor_id]
        for lang, sents in lang_dict.items():
            for idx, text in enumerate(sents):
                rows.append({
                    "sentence_id":    f"{factor_id}_ml_{lang}_{idx:02d}",
                    "factor_id":      factor_id,
                    "domain_id":      domain_id,
                    "surface_family": "F1",
                    "split":          "ml",
                    "language":       lang,
                    "sentence_text":  text,
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_ml_dataframe()
    print(f"Multilingual sentences: {len(df)}")
    print(df.groupby(["language", "domain_id"]).size().unstack(fill_value=0).to_string())
    print(f"\nFactors: {df['factor_id'].nunique()}  |  Languages: {df['language'].nunique()}")
