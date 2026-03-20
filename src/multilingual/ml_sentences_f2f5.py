"""
ml_sentences_f2f5.py — Multilingual F2–F5 sentences for semantic convergence experiment.

Provides accurate translations of the F2, F3 (first 2 only), F4, and F5 sentences
(2 per family per factor) for all 24 factors into German (de), Spanish (es),
Arabic MSA (ar), and Chinese Simplified (zh).

24 factors × 4 families × 4 languages × 2 sentences = 768 sentences total.
"""

ML_SENTENCES_F2F5 = {

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN A — Scalar temperature
    # ══════════════════════════════════════════════════════════════════════════

    "A1": {
        "F2": {
            "de": [
                "Der Gegenstand hat eine hohe Temperatur.",
                "Das Objekt weist eine intensive Hitze auf.",
            ],
            "es": [
                "El artículo tiene una temperatura alta.",
                "El objeto exhibe un calor intenso.",
            ],
            "ar": [
                "يتمتع الجسم بدرجة حرارة مرتفعة.",
                "يُظهر الجسم حرارة شديدة.",
            ],
            "zh": [
                "这件物品温度很高。",
                "这个物体散发着强烈的热量。",
            ],
        },
        "F3": {
            "de": [
                "Was das Objekt kennzeichnet, ist seine hohe Temperatur.",
                "Heiß ist es, wie man das Objekt beschreiben würde.",
            ],
            "es": [
                "Lo que caracteriza al objeto es su alta temperatura.",
                "Caliente es como se describiría el objeto.",
            ],
            "ar": [
                "ما يميّز هذا الجسم هو درجة حرارته المرتفعة.",
                "حارّ هو الوصف الذي يُطلق على هذا الجسم.",
            ],
            "zh": [
                "这个物体的特征是它的高温。",
                "用\u201c热\u201d来描述这个物体最为恰当。",
            ],
        },
        "F4": {
            "de": [
                "Eine Berührung würde sofort ein unangenehmes Hitzegefühl verursachen.",
                "Es strahlt genug Hitze aus, um sehr unangenehm zu sein.",
            ],
            "es": [
                "Tocarlo causaría una incomodidad inmediata por el calor.",
                "Irradia suficiente calor como para resultar muy incómodo.",
            ],
            "ar": [
                "لمسه سيسبب انزعاجاً فورياً بسبب الحرارة.",
                "يشعّ حرارة كافية ليكون مزعجاً للغاية.",
            ],
            "zh": [
                "触碰它会立即因高温而感到不适。",
                "它散发出的热量足以令人非常不舒服。",
            ],
        },
        "F5": {
            "de": [
                "Bei Berührung bemerkt man sofort die intensive Hitze.",
                "Wenn man es berührt, fühlt sich die Oberfläche spürbar heiß an.",
            ],
            "es": [
                "Al tocarlo, uno nota inmediatamente el calor intenso.",
                "Cuando se toca, la superficie se siente notablemente caliente.",
            ],
            "ar": [
                "عند اللمس، يلاحظ المرء فوراً الحرارة الشديدة.",
                "عند لمسه، يبدو السطح حاراً بشكل ملحوظ.",
            ],
            "zh": [
                "一接触就能立即感受到强烈的热度。",
                "触摸时，表面明显感觉很热。",
            ],
        },
    },

    "A2": {
        "F2": {
            "de": [
                "Der Gegenstand hat eine niedrige Temperatur.",
                "Das Objekt weist eine kalte Temperatur auf.",
            ],
            "es": [
                "El artículo tiene una temperatura baja.",
                "El objeto registra una temperatura fría.",
            ],
            "ar": [
                "يتمتع الجسم بدرجة حرارة منخفضة.",
                "يسجّل الجسم درجة حرارة باردة.",
            ],
            "zh": [
                "这件物品温度很低。",
                "这个物体呈现出低温。",
            ],
        },
        "F3": {
            "de": [
                "Kalt ist es, wie man das Objekt beschreiben würde.",
                "Was das Objekt kennzeichnet, ist seine niedrige Temperatur.",
            ],
            "es": [
                "Frío es como se describiría el objeto.",
                "Lo que caracteriza al objeto es su baja temperatura.",
            ],
            "ar": [
                "بارد هو الوصف الذي يُطلق على هذا الجسم.",
                "ما يميّز هذا الجسم هو درجة حرارته المنخفضة.",
            ],
            "zh": [
                "用\u201c冷\u201d来描述这个物体最为恰当。",
                "这个物体的特征是它的低温。",
            ],
        },
        "F4": {
            "de": [
                "Eine Berührung erzeugt ein starkes Kältegefühl.",
                "Es entzieht der Hand bei Kontakt die Wärme.",
            ],
            "es": [
                "Tocarlo produce una fuerte sensación de frío.",
                "Al contacto, extrae el calor de la mano.",
            ],
            "ar": [
                "لمسه يُنتج إحساساً قوياً بالبرودة.",
                "يسحب الحرارة من اليد عند التلامس.",
            ],
            "zh": [
                "触碰它会产生强烈的冰冷感。",
                "接触时，它会吸走手上的热量。",
            ],
        },
        "F5": {
            "de": [
                "Bei Berührung ist die Kälte sofort spürbar.",
                "Wenn man es berührt, sendet die Oberfläche einen Kälteschauer durch die Finger.",
            ],
            "es": [
                "Al tocarlo, el frío es inmediatamente evidente.",
                "Cuando se toca, la superficie envía un escalofrío por los dedos.",
            ],
            "ar": [
                "عند اللمس، تكون البرودة واضحة على الفور.",
                "عند لمسه، يبعث السطح برعشة برد عبر الأصابع.",
            ],
            "zh": [
                "一接触就能立即感受到冰冷。",
                "触摸时，表面的寒意直透指尖。",
            ],
        },
    },

    "A3": {
        "F2": {
            "de": [
                "Der Gegenstand hat eine mäßige Wärme.",
                "Das Objekt hat eine angenehm erhöhte Temperatur.",
            ],
            "es": [
                "El artículo tiene una tibieza moderada.",
                "El objeto tiene una temperatura agradablemente elevada.",
            ],
            "ar": [
                "يتمتع الجسم بدفء معتدل.",
                "يتمتع الجسم بدرجة حرارة مرتفعة بشكل مريح.",
            ],
            "zh": [
                "这件物品有着适度的温热。",
                "这个物体温度宜人地偏高。",
            ],
        },
        "F3": {
            "de": [
                "Warm ist es, wie man das Objekt beschreiben würde.",
                "Was das Objekt kennzeichnet, ist seine sanfte Wärme.",
            ],
            "es": [
                "Tibio es como se describiría el objeto.",
                "Lo que caracteriza al objeto es su suave calidez.",
            ],
            "ar": [
                "دافئ هو الوصف الذي يُطلق على هذا الجسم.",
                "ما يميّز هذا الجسم هو دفؤه اللطيف.",
            ],
            "zh": [
                "用\u201c温暖\u201d来描述这个物体最为恰当。",
                "这个物体的特征是它柔和的暖意。",
            ],
        },
        "F4": {
            "de": [
                "Eine Berührung vermittelt ein angenehmes Wärmegefühl.",
                "Es gibt milde Wärme ab, ohne unangenehm zu sein.",
            ],
            "es": [
                "Tocarlo produce una agradable sensación de calidez.",
                "Emite un calor suave sin resultar incómodo.",
            ],
            "ar": [
                "لمسه يمنح إحساساً لطيفاً بالدفء.",
                "يبعث حرارة خفيفة دون أن يكون مزعجاً.",
            ],
            "zh": [
                "触碰它会产生令人愉悦的温暖感。",
                "它散发出柔和的热量，并不令人不适。",
            ],
        },
        "F5": {
            "de": [
                "Bei Berührung spürt man eine sanfte, angenehme Wärme.",
                "Wenn man es berührt, fühlt es sich angenehm warm, aber nicht heiß an.",
            ],
            "es": [
                "Al tocarlo, uno siente una calidez suave y agradable.",
                "Cuando se toca, se siente agradablemente tibio pero no caliente.",
            ],
            "ar": [
                "عند اللمس، يشعر المرء بدفء لطيف ومريح.",
                "عند لمسه، يبدو دافئاً بشكل مريح لكنه ليس حاراً.",
            ],
            "zh": [
                "一接触就能感到柔和舒适的温暖。",
                "触摸时，感觉温暖宜人但并不烫手。",
            ],
        },
    },

    "A4": {
        "F2": {
            "de": [
                "Der Gegenstand liegt etwas unter Raumtemperatur.",
                "Das Objekt hat eine leicht niedrige Temperatur.",
            ],
            "es": [
                "El artículo está ligeramente por debajo de la temperatura ambiente.",
                "El objeto tiene una temperatura ligeramente baja.",
            ],
            "ar": [
                "يقع الجسم أدنى بقليل من درجة حرارة الغرفة.",
                "يتمتع الجسم بدرجة حرارة منخفضة قليلاً.",
            ],
            "zh": [
                "这件物品略低于室温。",
                "这个物体温度稍微偏低。",
            ],
        },
        "F3": {
            "de": [
                "Kühl ist es, wie man das Objekt beschreiben würde.",
                "Was das Objekt kennzeichnet, ist seine leichte Kühle.",
            ],
            "es": [
                "Fresco es como se describiría el objeto.",
                "Lo que caracteriza al objeto es su leve frescor.",
            ],
            "ar": [
                "فاتر هو الوصف الذي يُطلق على هذا الجسم.",
                "ما يميّز هذا الجسم هو برودته الخفيفة.",
            ],
            "zh": [
                "用\u201c凉\u201d来描述这个物体最为恰当。",
                "这个物体的特征是它微微的凉意。",
            ],
        },
        "F4": {
            "de": [
                "Eine Berührung erzeugt ein leichtes Kühlgefühl.",
                "Es fühlt sich kühler an als die umgebende Luft.",
            ],
            "es": [
                "Tocarlo produce una leve sensación de frescor.",
                "Se siente más fresco que el aire circundante.",
            ],
            "ar": [
                "لمسه يُنتج إحساساً خفيفاً بالبرودة.",
                "يبدو أبرد من الهواء المحيط.",
            ],
            "zh": [
                "触碰它会产生轻微的凉爽感。",
                "它摸起来比周围的空气更凉。",
            ],
        },
        "F5": {
            "de": [
                "Bei Berührung bemerkt man eine milde, erfrischende Kühle.",
                "Wenn man es berührt, fühlt es sich leicht kühl, aber nicht kalt an.",
            ],
            "es": [
                "Al tocarlo, uno nota un frescor leve y refrescante.",
                "Cuando se toca, se siente ligeramente fresco sin estar frío.",
            ],
            "ar": [
                "عند اللمس، يلاحظ المرء برودة خفيفة ومنعشة.",
                "عند لمسه، يبدو بارداً قليلاً دون أن يكون شديد البرودة.",
            ],
            "zh": [
                "一接触就能感到轻微而清爽的凉意。",
                "触摸时，感觉微凉但并不冰冷。",
            ],
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN B — Scalar size
    # ══════════════════════════════════════════════════════════════════════════

    "B1": {
        "F2": {
            "de": [
                "Der Gegenstand ist von großer Größe.",
                "Das Objekt nimmt beträchtlichen Raum ein.",
            ],
            "es": [
                "El artículo es de gran tamaño.",
                "El objeto ocupa un espacio considerable.",
            ],
            "ar": [
                "الجسم كبير الحجم.",
                "يشغل الجسم حيزاً كبيراً.",
            ],
            "zh": [
                "这件物品体积很大。",
                "这个物体占据了相当大的空间。",
            ],
        },
        "F3": {
            "de": [
                "Groß ist es, wie man das Objekt beschreiben würde.",
                "Wofür das Objekt bekannt ist, ist seine Größe.",
            ],
            "es": [
                "Grande es como se describiría el objeto.",
                "Por lo que se conoce al objeto es por ser grande.",
            ],
            "ar": [
                "كبير هو الوصف الذي يُطلق على هذا الجسم.",
                "ما يُعرف به هذا الجسم هو كونه كبيراً.",
            ],
            "zh": [
                "用\u201c大\u201d来描述这个物体最为恰当。",
                "这个物体的显著特征是它很大。",
            ],
        },
        "F4": {
            "de": [
                "Es ist merklich größer als der Durchschnitt.",
                "Man würde es im Vergleich zu ähnlichen Objekten als groß bezeichnen.",
            ],
            "es": [
                "Es notablemente más grande que el promedio.",
                "Se lo calificaría como grande en comparación con objetos similares.",
            ],
            "ar": [
                "إنه أكبر بشكل ملحوظ من المعتاد.",
                "يمكن وصفه بأنه كبير مقارنة بالأجسام المماثلة.",
            ],
            "zh": [
                "它明显大于平均水平。",
                "与同类物体相比，你会说它很大。",
            ],
        },
        "F5": {
            "de": [
                "Neben einem typischen Objekt erscheint es recht groß.",
                "Man bemerkt sofort die große Größe, wenn man das Objekt betrachtet.",
            ],
            "es": [
                "Colocado junto a un objeto típico, parece bastante grande.",
                "Uno nota de inmediato el gran tamaño al mirar el objeto.",
            ],
            "ar": [
                "عند وضعه بجانب جسم عادي، يبدو كبيراً جداً.",
                "يلاحظ المرء فوراً الحجم الكبير عند النظر إلى الجسم.",
            ],
            "zh": [
                "放在普通物体旁边时，它显得相当大。",
                "一看到这个物体，就会立刻注意到它的大尺寸。",
            ],
        },
    },

    "B2": {
        "F2": {
            "de": [
                "Der Gegenstand ist von kleiner Größe.",
                "Das Objekt nimmt sehr wenig Raum ein.",
            ],
            "es": [
                "El artículo es de tamaño pequeño.",
                "El objeto ocupa muy poco espacio.",
            ],
            "ar": [
                "الجسم صغير الحجم.",
                "يشغل الجسم حيزاً ضئيلاً جداً.",
            ],
            "zh": [
                "这件物品体积很小。",
                "这个物体几乎不占什么空间。",
            ],
        },
        "F3": {
            "de": [
                "Klein ist es, wie man das Objekt beschreiben würde.",
                "Wofür das Objekt bekannt ist, ist seine geringe Größe.",
            ],
            "es": [
                "Pequeño es como se describiría el objeto.",
                "Por lo que se conoce al objeto es por ser pequeño.",
            ],
            "ar": [
                "صغير هو الوصف الذي يُطلق على هذا الجسم.",
                "ما يُعرف به هذا الجسم هو كونه صغيراً.",
            ],
            "zh": [
                "用\u201c小\u201d来描述这个物体最为恰当。",
                "这个物体的显著特征是它很小。",
            ],
        },
        "F4": {
            "de": [
                "Es ist merklich kleiner als der Durchschnitt.",
                "Man würde es im Vergleich zu ähnlichen Objekten als klein bezeichnen.",
            ],
            "es": [
                "Es notablemente más pequeño que el promedio.",
                "Se lo calificaría como pequeño en comparación con objetos similares.",
            ],
            "ar": [
                "إنه أصغر بشكل ملحوظ من المعتاد.",
                "يمكن وصفه بأنه صغير مقارنة بالأجسام المماثلة.",
            ],
            "zh": [
                "它明显小于平均水平。",
                "与同类物体相比，你会说它很小。",
            ],
        },
        "F5": {
            "de": [
                "Neben einem typischen Objekt erscheint es recht klein.",
                "Man bemerkt sofort die geringe Größe, wenn man das Objekt betrachtet.",
            ],
            "es": [
                "Colocado junto a un objeto típico, parece bastante pequeño.",
                "Uno nota de inmediato el tamaño pequeño al mirar el objeto.",
            ],
            "ar": [
                "عند وضعه بجانب جسم عادي، يبدو صغيراً جداً.",
                "يلاحظ المرء فوراً الحجم الصغير عند النظر إلى الجسم.",
            ],
            "zh": [
                "放在普通物体旁边时，它显得相当小。",
                "一看到这个物体，就会立刻注意到它的小尺寸。",
            ],
        },
    },

    "B3": {
        "F2": {
            "de": [
                "Der Gegenstand ist von enormer Größe.",
                "Das Objekt ist äußerst groß.",
            ],
            "es": [
                "El artículo es de tamaño enorme.",
                "El objeto es extremadamente grande.",
            ],
            "ar": [
                "الجسم هائل الحجم.",
                "الجسم كبير للغاية.",
            ],
            "zh": [
                "这件物品体积庞大。",
                "这个物体极其巨大。",
            ],
        },
        "F3": {
            "de": [
                "Riesig ist es, wie man das Objekt beschreiben würde.",
                "Wofür das Objekt bekannt ist, ist seine riesige Größe.",
            ],
            "es": [
                "Enorme es como se describiría el objeto.",
                "Por lo que se conoce al objeto es por ser enorme.",
            ],
            "ar": [
                "ضخم هو الوصف الذي يُطلق على هذا الجسم.",
                "ما يُعرف به هذا الجسم هو كونه ضخماً.",
            ],
            "zh": [
                "用\u201c巨大\u201d来描述这个物体最为恰当。",
                "这个物体的显著特征是它非常巨大。",
            ],
        },
        "F4": {
            "de": [
                "Es ist bei Weitem größer als vergleichbare Objekte.",
                "Nur wenige Dinge sind so groß wie dieses Objekt.",
            ],
            "es": [
                "Es con mucho más grande que objetos comparables.",
                "Pocas cosas son tan grandes como este objeto.",
            ],
            "ar": [
                "إنه أكبر بكثير من الأجسام المماثلة.",
                "قلّ ما يوجد شيء بحجم هذا الجسم.",
            ],
            "zh": [
                "它远比同类物体大得多。",
                "很少有东西能和这个物体一样大。",
            ],
        },
        "F5": {
            "de": [
                "Neben einem typischen Objekt ist es dramatisch größer.",
                "Die schiere Riesigkeit des Objekts ist sofort erkennbar.",
            ],
            "es": [
                "Colocado junto a un objeto típico, es dramáticamente más grande.",
                "La enorme magnitud del objeto es inmediatamente evidente.",
            ],
            "ar": [
                "عند وضعه بجانب جسم عادي، يبدو أكبر بشكل مذهل.",
                "الحجم الهائل للجسم واضح على الفور.",
            ],
            "zh": [
                "放在普通物体旁边时，它大得惊人。",
                "这个物体的庞大体积一眼就能看出。",
            ],
        },
    },

    "B4": {
        "F2": {
            "de": [
                "Der Gegenstand ist von winziger Größe.",
                "Das Objekt ist äußerst klein.",
            ],
            "es": [
                "El artículo es de tamaño minúsculo.",
                "El objeto es extremadamente pequeño.",
            ],
            "ar": [
                "الجسم متناهي الصغر.",
                "الجسم صغير للغاية.",
            ],
            "zh": [
                "这件物品体积极小。",
                "这个物体极其微小。",
            ],
        },
        "F3": {
            "de": [
                "Winzig ist es, wie man das Objekt beschreiben würde.",
                "Wofür das Objekt bekannt ist, ist seine winzige Größe.",
            ],
            "es": [
                "Diminuto es como se describiría el objeto.",
                "Por lo que se conoce al objeto es por ser diminuto.",
            ],
            "ar": [
                "ضئيل هو الوصف الذي يُطلق على هذا الجسم.",
                "ما يُعرف به هذا الجسم هو كونه ضئيلاً.",
            ],
            "zh": [
                "用\u201c微小\u201d来描述这个物体最为恰当。",
                "这个物体的显著特征是它非常微小。",
            ],
        },
        "F4": {
            "de": [
                "Es ist drastisch kleiner als vergleichbare Objekte.",
                "Nur wenige Dinge sind so klein wie dieses Objekt.",
            ],
            "es": [
                "Es drásticamente más pequeño que objetos comparables.",
                "Pocas cosas son tan pequeñas como este objeto.",
            ],
            "ar": [
                "إنه أصغر بكثير من الأجسام المماثلة.",
                "قلّ ما يوجد شيء بصغر حجم هذا الجسم.",
            ],
            "zh": [
                "它远比同类物体小得多。",
                "很少有东西能和这个物体一样小。",
            ],
        },
        "F5": {
            "de": [
                "Neben einem typischen Objekt erscheint es sehr winzig.",
                "Die winzige Größe des Objekts ist sofort erkennbar.",
            ],
            "es": [
                "Colocado junto a un objeto típico, parece muy diminuto.",
                "El diminuto tamaño del objeto es inmediatamente evidente.",
            ],
            "ar": [
                "عند وضعه بجانب جسم عادي، يبدو ضئيلاً جداً.",
                "الحجم الضئيل للجسم واضح على الفور.",
            ],
            "zh": [
                "放在普通物体旁边时，它显得非常微小。",
                "这个物体的微小尺寸一眼就能看出。",
            ],
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN C — Animal taxonomy
    # ══════════════════════════════════════════════════════════════════════════

    "C1": {
        "F2": {
            "de": [
                "Es ist ein domestizierter Canide.",
                "Das Tier gehört zur Hundeart.",
            ],
            "es": [
                "Es un canino domesticado.",
                "El animal pertenece a la especie de los perros.",
            ],
            "ar": [
                "إنه حيوان من فصيلة الكلبيات المستأنسة.",
                "ينتمي هذا الحيوان إلى نوع الكلاب.",
            ],
            "zh": [
                "它是一种驯化的犬类。",
                "这只动物属于犬种。",
            ],
        },
        "F3": {
            "de": [
                "Was das Tier genau ist, ist ein Hund.",
                "Ein Hund ist das, was dieses Tier ist.",
            ],
            "es": [
                "Lo que este animal es, específicamente, es un perro.",
                "Un perro es lo que este animal es.",
            ],
            "ar": [
                "ما هذا الحيوان تحديداً هو كلب.",
                "كلب هو ما يكون عليه هذا الحيوان.",
            ],
            "zh": [
                "这只动物确切来说是一只狗。",
                "狗就是这只动物的身份。",
            ],
        },
        "F4": {
            "de": [
                "Dies ist die Art von Tier, die bellt und als Haustier gehalten wird.",
                "Es ist der häufigste vierbeinige Begleiter im Haushalt.",
            ],
            "es": [
                "Este es el tipo de animal que ladra y se tiene como mascota.",
                "Es el compañero canino doméstico más común.",
            ],
            "ar": [
                "هذا هو نوع الحيوان الذي ينبح ويُربّى كحيوان أليف.",
                "إنه أكثر الرفاق المنزلية من فصيلة الكلبيات شيوعاً.",
            ],
            "zh": [
                "这种动物会吠叫，通常被人们作为宠物饲养。",
                "它是最常见的家庭犬科伴侣。",
            ],
        },
        "F5": {
            "de": [
                "Als Hund ist dieses Tier eines der bekanntesten Haustiere.",
                "Dieses Tier gehört zur Art, die allgemein als Haushund bekannt ist.",
            ],
            "es": [
                "Como perro, este animal es una de las mascotas domésticas más conocidas.",
                "Este animal pertenece a la especie comúnmente conocida como perro doméstico.",
            ],
            "ar": [
                "بوصفه كلباً، يُعدّ هذا الحيوان من أشهر الحيوانات الأليفة.",
                "ينتمي هذا الحيوان إلى النوع المعروف عادةً بالكلب المنزلي.",
            ],
            "zh": [
                "作为一只狗，这只动物是最常见的家养宠物之一。",
                "这只动物属于通常被称为家犬的物种。",
            ],
        },
    },

    "C2": {
        "F2": {
            "de": [
                "Es gehört zur Familie der Hunde.",
                "Das Tier ist ein Mitglied der Familie der Caniden.",
            ],
            "es": [
                "Pertenece a la familia de los perros.",
                "El animal es un miembro de la familia de los cánidos.",
            ],
            "ar": [
                "ينتمي إلى فصيلة الكلبيات.",
                "هذا الحيوان عضو في فصيلة الكلبيات.",
            ],
            "zh": [
                "它属于犬科家族。",
                "这只动物是犬科家族的成员。",
            ],
        },
        "F3": {
            "de": [
                "Was das Tier genau ist, ist ein Canide.",
                "Ein Canide ist das, was dieses Tier ist.",
            ],
            "es": [
                "Lo que este animal es, específicamente, es un cánido.",
                "Un cánido es lo que este animal es.",
            ],
            "ar": [
                "ما هذا الحيوان تحديداً هو حيوان من فصيلة الكلبيات.",
                "حيوان من الكلبيات هو ما يكون عليه هذا الحيوان.",
            ],
            "zh": [
                "这只动物确切来说是犬科动物。",
                "犬科动物就是这只动物的分类。",
            ],
        },
        "F4": {
            "de": [
                "Dies ist ein Tier aus der Hundefamilie, breiter als nur Hunde.",
                "Es gehört zur taxonomischen Gruppe, die Wölfe und Füchse umfasst.",
            ],
            "es": [
                "Este es un animal de la familia de los perros, más amplia que solo los perros.",
                "Es miembro del grupo taxonómico que incluye lobos y zorros.",
            ],
            "ar": [
                "هذا حيوان من فصيلة الكلبيات، وهي أوسع من مجرد الكلاب.",
                "إنه عضو في المجموعة التصنيفية التي تشمل الذئاب والثعالب.",
            ],
            "zh": [
                "这是一种犬科动物，其范围比单纯的狗更广。",
                "它是包括狼和狐狸在内的分类群的成员。",
            ],
        },
        "F5": {
            "de": [
                "Als Canide gehört dieses Tier zur selben Familie wie Wölfe.",
                "Dieses Tier wird innerhalb der Canidenfamilie klassifiziert, was es zu einem Caniden macht.",
            ],
            "es": [
                "Como cánido, este animal pertenece a la misma familia que los lobos.",
                "Este animal se clasifica dentro de la familia de los cánidos, lo que lo convierte en un cánido.",
            ],
            "ar": [
                "بوصفه من الكلبيات، ينتمي هذا الحيوان إلى نفس فصيلة الذئاب.",
                "يُصنَّف هذا الحيوان ضمن فصيلة الكلبيات، مما يجعله من الكلبيات.",
            ],
            "zh": [
                "作为犬科动物，这只动物与狼同属一个科。",
                "这只动物被归类为犬科，因此它是犬科动物。",
            ],
        },
    },

    "C3": {
        "F2": {
            "de": [
                "Es gehört zur Klasse der Säugetiere.",
                "Das Tier ist ein Säugetierwesen.",
            ],
            "es": [
                "Pertenece a la clase de los mamíferos.",
                "El animal es una criatura mamífera.",
            ],
            "ar": [
                "ينتمي إلى طائفة الثدييات.",
                "هذا الحيوان كائن ثديي.",
            ],
            "zh": [
                "它属于哺乳动物纲。",
                "这只动物是哺乳类生物。",
            ],
        },
        "F3": {
            "de": [
                "Was das Tier genau ist, ist ein Säugetier.",
                "Ein Säugetier ist das, was dieses Tier ist.",
            ],
            "es": [
                "Lo que este animal es, específicamente, es un mamífero.",
                "Un mamífero es lo que este animal es.",
            ],
            "ar": [
                "ما هذا الحيوان تحديداً هو حيوان ثديي.",
                "حيوان ثديي هو ما يكون عليه هذا الحيوان.",
            ],
            "zh": [
                "这只动物确切来说是哺乳动物。",
                "哺乳动物就是这只动物的分类。",
            ],
        },
        "F4": {
            "de": [
                "Dieses Tier ist warmblütig und säugt seine Jungen mit Milch.",
                "Es ist die Art von Tier, die Haare hat und Junge mit Milch ernährt.",
            ],
            "es": [
                "Este animal es de sangre caliente y amamanta a sus crías con leche.",
                "Es el tipo de animal que tiene pelo y alimenta a sus crías con leche.",
            ],
            "ar": [
                "هذا الحيوان ذو دم حار ويُرضع صغاره بالحليب.",
                "إنه نوع الحيوان الذي يمتلك شعراً ويُغذّي صغاره بالحليب.",
            ],
            "zh": [
                "这只动物是温血的，用乳汁哺育幼崽。",
                "它是那种有毛发并用乳汁喂养幼崽的动物。",
            ],
        },
        "F5": {
            "de": [
                "Als Säugetier ist dieses Tier warmblütig und zieht Junge mit Milch auf.",
                "Dieses Lebewesen gehört zur Klasse der Tiere, die Menschen und Wale umfasst.",
            ],
            "es": [
                "Como mamífero, este animal es de sangre caliente y cría a sus jóvenes con leche.",
                "Esta criatura pertenece a la clase de animales que incluye a los humanos y las ballenas.",
            ],
            "ar": [
                "بوصفه من الثدييات، هذا الحيوان ذو دم حار ويربّي صغاره بالحليب.",
                "ينتمي هذا المخلوق إلى طائفة الحيوانات التي تشمل البشر والحيتان.",
            ],
            "zh": [
                "作为哺乳动物，这只动物是温血的，用乳汁抚养幼崽。",
                "这个生物属于包括人类和鲸鱼在内的动物纲。",
            ],
        },
    },

    "C4": {
        "F2": {
            "de": [
                "Er ist darauf trainiert, Aufgaben für Menschen auszuführen.",
                "Das Tier ist ein für Aufgaben ausgebildeter Hund.",
            ],
            "es": [
                "Está entrenado para realizar tareas para personas.",
                "El animal es un perro entrenado para tareas.",
            ],
            "ar": [
                "إنه مدرّب على أداء مهام للبشر.",
                "هذا الحيوان كلب مدرّب على المهام.",
            ],
            "zh": [
                "它经过训练，为人类执行任务。",
                "这只动物是一只经过任务训练的狗。",
            ],
        },
        "F3": {
            "de": [
                "Was diesen Hund ausmacht, ist seine Arbeitsrolle.",
                "Ein Arbeitshund ist das, was dieser Hund ist.",
            ],
            "es": [
                "Lo que define a este perro es su rol de trabajo.",
                "Un perro de trabajo es lo que este perro es.",
            ],
            "ar": [
                "ما يُحدّد هذا الكلب هو دوره في العمل.",
                "كلب عمل هو ما يكون عليه هذا الكلب.",
            ],
            "zh": [
                "这只狗的定义特征是它的工作角色。",
                "工作犬就是这只狗的身份。",
            ],
        },
        "F4": {
            "de": [
                "Dieser Hund ist für bestimmte Aufgaben ausgebildet, nicht nur zur Gesellschaft.",
                "Er erfüllt Funktionen wie Hüten, Bewachen oder das Unterstützen von Menschen.",
            ],
            "es": [
                "Este perro está entrenado para tareas específicas, no solo como compañía.",
                "Desempeña funciones como pastoreo, vigilancia o asistencia a personas.",
            ],
            "ar": [
                "هذا الكلب مدرّب على مهام محددة وليس فقط للرفقة.",
                "يؤدي وظائف مثل الرعي أو الحراسة أو مساعدة البشر.",
            ],
            "zh": [
                "这只狗受过专门训练来执行任务，而不仅仅是陪伴。",
                "它执行放牧、看守或协助人类等功能。",
            ],
        },
        "F5": {
            "de": [
                "Als Arbeitshund ist er für bestimmte praktische Aufgaben trainiert und eingesetzt.",
                "Im Gegensatz zu Haushunden dient dieser Hund in einer beruflichen oder dienstlichen Funktion.",
            ],
            "es": [
                "Como perro de trabajo, está entrenado y utilizado para tareas prácticas específicas.",
                "A diferencia de los perros de compañía, este perro sirve en una capacidad profesional o de servicio.",
            ],
            "ar": [
                "بوصفه كلب عمل، فهو مدرّب ومُستخدم لمهام عملية محددة.",
                "على عكس كلاب الرفقة، يخدم هذا الكلب في دور مهني أو خدمي.",
            ],
            "zh": [
                "作为工作犬，它经过训练并被用于特定的实际任务。",
                "与宠物犬不同，这只狗在专业或服务岗位上工作。",
            ],
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN D — Vehicles and software
    # ══════════════════════════════════════════════════════════════════════════

    "D1": {
        "F2": {
            "de": [
                "Es ist ein straßentaugliches Passagierfahrzeug mit Elektroantrieb.",
                "Das Auto befördert Personen und wird mit Batteriestrom betrieben.",
            ],
            "es": [
                "Es un vehículo de pasajeros de carretera impulsado por electricidad.",
                "El automóvil transporta personas y funciona con energía de batería.",
            ],
            "ar": [
                "إنها مركبة ركاب تسير على الطريق وتعمل بالكهرباء.",
                "تنقل هذه السيارة الركاب وتعمل بطاقة البطارية.",
            ],
            "zh": [
                "它是一辆以电力驱动的公路载客车辆。",
                "这辆车载运乘客，以电池供电。",
            ],
        },
        "F3": {
            "de": [
                "Was dieses Fahrzeug ausmacht, ist, dass es elektrisch ist und Passagiere befördert.",
                "Ein elektrisches Passagierfahrzeug ist das, was dies ist.",
            ],
            "es": [
                "Lo que define a este vehículo es que es eléctrico y transporta pasajeros.",
                "Un vehículo eléctrico de pasajeros es lo que esto es.",
            ],
            "ar": [
                "ما يُحدّد هذه المركبة هو أنها كهربائية وتنقل الركاب.",
                "مركبة ركاب كهربائية هي ما تكون عليه هذه المركبة.",
            ],
            "zh": [
                "这辆车的定义特征是它是电动的并且载运乘客。",
                "电动载客车辆就是它的身份。",
            ],
        },
        "F4": {
            "de": [
                "Dieses Fahrzeug wird mit Strom betrieben und ist dafür konzipiert, Personen zu befördern.",
                "Es ist ein batteriebetriebenes Auto zur Beförderung von Passagieren.",
            ],
            "es": [
                "Este vehículo funciona con electricidad y está diseñado para transportar personas.",
                "Es un automóvil impulsado por batería destinado al transporte de pasajeros.",
            ],
            "ar": [
                "تعمل هذه المركبة بالكهرباء وهي مصممة لنقل الأشخاص.",
                "إنها سيارة تعمل بالبطارية مخصصة لنقل الركاب.",
            ],
            "zh": [
                "这辆车以电力运行，设计用来运送乘客。",
                "它是一辆以电池驱动、用于载客的汽车。",
            ],
        },
        "F5": {
            "de": [
                "Als elektrisches Passagierfahrzeug befördert es Personen mit batteriebetriebenen Motoren.",
                "Dieses Fahrzeug transportiert Passagiere auf Straßen mithilfe gespeicherter elektrischer Energie.",
            ],
            "es": [
                "Como vehículo eléctrico de pasajeros, transporta personas mediante motores alimentados por batería.",
                "Este vehículo transporta pasajeros por carretera utilizando energía eléctrica almacenada.",
            ],
            "ar": [
                "بوصفها مركبة ركاب كهربائية، تنقل الأشخاص باستخدام محركات تعمل بالبطارية.",
                "تنقل هذه المركبة الركاب على الطرق باستخدام الطاقة الكهربائية المخزنة.",
            ],
            "zh": [
                "作为电动载客车辆，它使用电池驱动的电机运送乘客。",
                "这辆车利用储存的电能在公路上运送乘客。",
            ],
        },
    },

    "D2": {
        "F2": {
            "de": [
                "Es ist dafür bestimmt, Güter statt Passagiere zu transportieren.",
                "Das Fahrzeug ist für den Gütertransport ausgelegt.",
            ],
            "es": [
                "Está destinado a transportar mercancías en lugar de pasajeros.",
                "El vehículo está diseñado para transportar carga.",
            ],
            "ar": [
                "إنها مخصصة لنقل البضائع بدلاً من الركاب.",
                "المركبة مصممة لنقل الشحنات.",
            ],
            "zh": [
                "它用于运载货物而非乘客。",
                "这辆车是为运输货物而设计的。",
            ],
        },
        "F3": {
            "de": [
                "Was dieses Fahrzeug ausmacht, ist sein Zweck als Gütertransporter.",
                "Ein Güterfahrzeug ist das, was dies ist.",
            ],
            "es": [
                "Lo que define a este vehículo es su función de transporte de carga.",
                "Un vehículo de carga es lo que esto es.",
            ],
            "ar": [
                "ما يُحدّد هذه المركبة هو غرضها في نقل البضائع.",
                "مركبة شحن هي ما تكون عليه هذه المركبة.",
            ],
            "zh": [
                "这辆车的定义特征是它的货运用途。",
                "货运车辆就是它的身份。",
            ],
        },
        "F4": {
            "de": [
                "Dieses Fahrzeug ist gebaut, um Fracht statt Personen zu transportieren.",
                "Es ist ein Fahrzeugtyp, der dazu bestimmt ist, Waren von Ort zu Ort zu bringen.",
            ],
            "es": [
                "Este vehículo está construido para mover mercancías en lugar de personas.",
                "Es un tipo de vehículo destinado a transportar mercancías de un lugar a otro.",
            ],
            "ar": [
                "بُنيت هذه المركبة لنقل الشحنات بدلاً من الأشخاص.",
                "إنها نوع من المركبات مخصص لنقل البضائع من مكان إلى آخر.",
            ],
            "zh": [
                "这辆车是为运输货物而非人员而建造的。",
                "它是一种用于将货物从一处运到另一处的车辆。",
            ],
        },
        "F5": {
            "de": [
                "Als Güterfahrzeug ist es dafür ausgelegt, Waren statt Passagiere zu transportieren.",
                "Die Hauptfunktion dieses Fahrzeugs ist der Transport von Fracht und Gütern.",
            ],
            "es": [
                "Como vehículo de carga, está diseñado para mover mercancías en lugar de pasajeros.",
                "La función principal de este vehículo es el transporte de carga y mercancías.",
            ],
            "ar": [
                "بوصفها مركبة شحن، فهي مصممة لنقل البضائع بدلاً من الركاب.",
                "الوظيفة الأساسية لهذه المركبة هي نقل الشحنات والبضائع.",
            ],
            "zh": [
                "作为货运车辆，它的设计目的是运输货物而非乘客。",
                "这辆车的主要功能是运输货物和商品。",
            ],
        },
    },

    "D3": {
        "F2": {
            "de": [
                "Es ist ein kleiner Personenwagen.",
                "Das Auto gehört zur Kompaktklasse.",
            ],
            "es": [
                "Es un automóvil pequeño de pasajeros.",
                "El automóvil está construido en la categoría de tamaño compacto.",
            ],
            "ar": [
                "إنها سيارة ركاب صغيرة.",
                "السيارة مصنوعة في فئة الحجم المدمج.",
            ],
            "zh": [
                "它是一辆小型载客轿车。",
                "这辆车属于紧凑级别。",
            ],
        },
        "F3": {
            "de": [
                "Was dieses Fahrzeug ausmacht, ist, dass es ein Kompaktwagen ist.",
                "Ein Kompaktwagen ist das, was dieses Fahrzeug ist.",
            ],
            "es": [
                "Lo que define a este vehículo es ser un automóvil compacto.",
                "Un automóvil compacto es lo que este vehículo es.",
            ],
            "ar": [
                "ما يُحدّد هذه المركبة هو كونها سيارة مدمجة.",
                "سيارة مدمجة هي ما تكون عليه هذه المركبة.",
            ],
            "zh": [
                "这辆车的定义特征是它是一辆紧凑型轿车。",
                "紧凑型轿车就是这辆车的类型。",
            ],
        },
        "F4": {
            "de": [
                "Dieses Auto ist deutlich kleiner als ein normales Fahrzeug in voller Größe.",
                "Es ist ein kleines Personenfahrzeug, effizient für den Stadtverkehr.",
            ],
            "es": [
                "Este automóvil es notablemente más pequeño que un vehículo estándar de tamaño completo.",
                "Es un automóvil pequeño de pasajeros, eficiente para uso urbano.",
            ],
            "ar": [
                "هذه السيارة أصغر بشكل ملحوظ من سيارة عادية كاملة الحجم.",
                "إنها سيارة ركاب صغيرة وفعّالة للاستخدام في المدن.",
            ],
            "zh": [
                "这辆车明显比标准全尺寸车辆要小。",
                "它是一辆适合城市使用的小型载客汽车。",
            ],
        },
        "F5": {
            "de": [
                "Als Kompaktwagen nimmt es weniger Platz ein als herkömmliche Limousinen.",
                "Dieses Auto wird aufgrund seiner relativ geringen Größe als kompakt eingestuft.",
            ],
            "es": [
                "Como automóvil compacto, ocupa menos espacio que los sedanes estándar.",
                "Este automóvil se clasifica como compacto debido a su tamaño relativamente pequeño.",
            ],
            "ar": [
                "بوصفها سيارة مدمجة، تشغل مساحة أقل من سيارات السيدان العادية.",
                "تُصنَّف هذه السيارة كمدمجة نظراً لحجمها الصغير نسبياً.",
            ],
            "zh": [
                "作为紧凑型轿车，它比标准轿车占用更少的空间。",
                "这辆车因其相对较小的尺寸而被归类为紧凑型。",
            ],
        },
    },

    "D4": {
        "F2": {
            "de": [
                "Es speichert Daten auf mehreren Maschinen.",
                "Das Datenbanksystem ist über mehrere Knoten verteilt.",
            ],
            "es": [
                "Almacena datos en múltiples máquinas.",
                "El sistema de base de datos está distribuido en varios nodos.",
            ],
            "ar": [
                "يخزّن البيانات عبر عدة أجهزة.",
                "نظام قاعدة البيانات موزّع على عدة عقد.",
            ],
            "zh": [
                "它将数据存储在多台机器上。",
                "该数据库系统分布在多个节点上。",
            ],
        },
        "F3": {
            "de": [
                "Was dieses System ausmacht, ist, dass es eine verteilte Datenbank ist.",
                "Eine verteilte Datenbank ist das, was diese Software ist.",
            ],
            "es": [
                "Lo que define a este sistema es ser una base de datos distribuida.",
                "Una base de datos distribuida es lo que este software es.",
            ],
            "ar": [
                "ما يُحدّد هذا النظام هو كونه قاعدة بيانات موزعة.",
                "قاعدة بيانات موزعة هي ما يكون عليه هذا البرنامج.",
            ],
            "zh": [
                "这个系统的定义特征是它是一个分布式数据库。",
                "分布式数据库就是这个软件的类型。",
            ],
        },
        "F4": {
            "de": [
                "Diese Software speichert und verwaltet Daten auf mehreren Servern.",
                "Es ist eine Datenbank, die nicht auf eine einzige Maschine zur Speicherung angewiesen ist.",
            ],
            "es": [
                "Este software almacena y gestiona datos en múltiples servidores.",
                "Es una base de datos que no depende de una sola máquina para el almacenamiento.",
            ],
            "ar": [
                "يخزّن هذا البرنامج البيانات ويديرها عبر عدة خوادم.",
                "إنها قاعدة بيانات لا تعتمد على جهاز واحد للتخزين.",
            ],
            "zh": [
                "这个软件在多台服务器上存储和管理数据。",
                "它是一个不依赖单台机器进行存储的数据库。",
            ],
        },
        "F5": {
            "de": [
                "Als verteilte Datenbank verteilt sie Daten auf viele Knoten für Zuverlässigkeit.",
                "Diese Datenbank verwaltet Daten über einen Cluster von Maschinen statt über einen einzelnen Server.",
            ],
            "es": [
                "Como base de datos distribuida, reparte los datos en muchos nodos para mayor fiabilidad.",
                "Esta base de datos gestiona datos a través de un clúster de máquinas en lugar de un solo servidor.",
            ],
            "ar": [
                "بوصفها قاعدة بيانات موزعة، توزّع البيانات على عدة عقد لضمان الموثوقية.",
                "تدير قاعدة البيانات هذه البيانات عبر مجموعة من الأجهزة بدلاً من خادم واحد.",
            ],
            "zh": [
                "作为分布式数据库，它将数据分散到多个节点以保证可靠性。",
                "这个数据库跨多台机器的集群管理数据，而非依赖单一服务器。",
            ],
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN E — Designed objects
    # ══════════════════════════════════════════════════════════════════════════

    "E1": {
        "F2": {
            "de": [
                "Es ist ein Wanderstiefel, der Wasser fernhält.",
                "Der Bergstiefel ist wasserabweisend.",
            ],
            "es": [
                "Es una bota de senderismo que impide la entrada de agua.",
                "La bota de montaña resiste la entrada de agua.",
            ],
            "ar": [
                "إنه حذاء مشي يمنع دخول الماء.",
                "حذاء المشي الجبلي يقاوم دخول الماء.",
            ],
            "zh": [
                "它是一双能防水的登山靴。",
                "这双越野靴可以抵御水的渗入。",
            ],
        },
        "F3": {
            "de": [
                "Was dieses Schuhwerk ausmacht, ist, dass es ein wasserdichter Wanderstiefel ist.",
                "Ein wasserdichter Wanderstiefel ist das, was dies ist.",
            ],
            "es": [
                "Lo que define este calzado es ser una bota de senderismo impermeable.",
                "Una bota de senderismo impermeable es lo que esto es.",
            ],
            "ar": [
                "ما يُحدّد هذا الحذاء هو كونه حذاء مشي مقاوماً للماء.",
                "حذاء مشي مقاوم للماء هو ما يكون عليه هذا الحذاء.",
            ],
            "zh": [
                "这双鞋的定义特征是它是一双防水登山靴。",
                "防水登山靴就是它的类型。",
            ],
        },
        "F4": {
            "de": [
                "Dieser Stiefel ist zum Wandern auf Wegen gemacht und hält die Füße dabei trocken.",
                "Es ist ein Outdoor-Schuh, der für nasse Wanderbedingungen konzipiert ist.",
            ],
            "es": [
                "Esta bota está hecha para caminar por senderos manteniendo los pies secos.",
                "Es un calzado de exterior diseñado para condiciones de senderismo húmedas.",
            ],
            "ar": [
                "هذا الحذاء مصنوع للمشي في المسارات مع إبقاء القدمين جافتين.",
                "إنه حذاء خارجي مصمم لظروف المشي الرطبة.",
            ],
            "zh": [
                "这双靴子是为在步道上行走同时保持双脚干燥而制造的。",
                "它是一种专为潮湿登山条件设计的户外鞋。",
            ],
        },
        "F5": {
            "de": [
                "Als wasserdichter Wanderstiefel ist er für den Einsatz auf Wegen in nasser Umgebung gebaut.",
                "Dieser Stiefel schützt die Füße vor Feuchtigkeit und bietet gleichzeitig Halt beim Wandern.",
            ],
            "es": [
                "Como bota de senderismo impermeable, está construida para uso en senderos en ambientes húmedos.",
                "Esta bota protege los pies de la humedad mientras proporciona soporte para el senderismo.",
            ],
            "ar": [
                "بوصفه حذاء مشي مقاوماً للماء، فهو مصنوع للاستخدام في المسارات في البيئات الرطبة.",
                "يحمي هذا الحذاء القدمين من الرطوبة مع توفير الدعم أثناء المشي.",
            ],
            "zh": [
                "作为防水登山靴，它专为在潮湿环境中的步道使用而制造。",
                "这双靴子在为登山提供支撑的同时保护双脚免受潮湿。",
            ],
        },
    },

    "E2": {
        "F2": {
            "de": [
                "Es ist ein Bürostuhl, der auf ergonomischen Halt ausgelegt ist.",
                "Der Stuhl ist für Schreibtischarbeit und körperlichen Komfort gefertigt.",
            ],
            "es": [
                "Es una silla de oficina diseñada para soporte ergonómico.",
                "La silla está hecha para trabajo de escritorio y comodidad corporal.",
            ],
            "ar": [
                "إنه كرسي مكتبي مصمم لدعم مريح للجسم.",
                "الكرسي مصنوع للعمل المكتبي وراحة الجسم.",
            ],
            "zh": [
                "它是一把为人体工程学支撑而设计的办公椅。",
                "这把椅子是为办公桌工作和身体舒适而制作的。",
            ],
        },
        "F3": {
            "de": [
                "Was diesen Stuhl ausmacht, ist, dass er ergonomisch und für das Büro bestimmt ist.",
                "Ein ergonomischer Bürostuhl ist das, was dies ist.",
            ],
            "es": [
                "Lo que define esta silla es ser ergonómica y para uso de oficina.",
                "Una silla de oficina ergonómica es lo que esto es.",
            ],
            "ar": [
                "ما يُحدّد هذا الكرسي هو كونه مريحاً للجسم ومخصصاً للمكتب.",
                "كرسي مكتبي مريح للجسم هو ما يكون عليه هذا الكرسي.",
            ],
            "zh": [
                "这把椅子的定义特征是它符合人体工程学且用于办公。",
                "人体工程学办公椅就是它的类型。",
            ],
        },
        "F4": {
            "de": [
                "Dieser Stuhl ist so konstruiert, dass er die Körperhaltung bei langen Stunden am Schreibtisch unterstützt.",
                "Es ist ein Arbeitssitz, der darauf ausgelegt ist, körperliche Belastung zu reduzieren.",
            ],
            "es": [
                "Esta silla está diseñada para apoyar la postura durante largas horas en un escritorio.",
                "Es un asiento de trabajo diseñado para reducir la tensión física.",
            ],
            "ar": [
                "هذا الكرسي مصمم لدعم وضعية الجسم خلال ساعات طويلة أمام المكتب.",
                "إنه مقعد عمل مصمم لتقليل الإجهاد البدني.",
            ],
            "zh": [
                "这把椅子旨在支撑长时间伏案工作时的身体姿势。",
                "它是一把为减少身体疲劳而设计的工作座椅。",
            ],
        },
        "F5": {
            "de": [
                "Als ergonomischer Bürostuhl unterstützt er den Körper bei der Schreibtischarbeit.",
                "Dieser Stuhl bietet ergonomischen Halt, der für Büroumgebungen geeignet ist.",
            ],
            "es": [
                "Como silla de oficina ergonómica, apoya el cuerpo durante el trabajo de escritorio.",
                "Esta silla proporciona soporte ergonómico adecuado para entornos de oficina.",
            ],
            "ar": [
                "بوصفه كرسي مكتب مريحاً للجسم، يدعم الجسم أثناء العمل المكتبي.",
                "يوفّر هذا الكرسي دعماً مريحاً للجسم يناسب بيئات المكاتب.",
            ],
            "zh": [
                "作为人体工程学办公椅，它在办公桌工作时为身体提供支撑。",
                "这把椅子提供适合办公环境的人体工程学支撑。",
            ],
        },
    },

    "E3": {
        "F2": {
            "de": [
                "Es ist ein Gebäude, das Wohnungen für Menschen enthält.",
                "Das Gebäude dient der Unterbringung mehrerer Haushalte.",
            ],
            "es": [
                "Es una estructura que contiene apartamentos para que la gente viva.",
                "El edificio se usa para alojar múltiples hogares.",
            ],
            "ar": [
                "إنه مبنى يحتوي على شقق ليعيش فيها الناس.",
                "يُستخدم المبنى لإيواء أسر متعددة.",
            ],
            "zh": [
                "它是一栋包含供人居住的公寓的建筑。",
                "这栋建筑用于容纳多个家庭。",
            ],
        },
        "F3": {
            "de": [
                "Was dieses Gebäude ausmacht, ist, dass es ein Wohnhaus mit Apartments ist.",
                "Ein Wohnhaus mit Apartments ist das, was dieses Gebäude ist.",
            ],
            "es": [
                "Lo que define este edificio es ser un edificio residencial de apartamentos.",
                "Un edificio residencial de apartamentos es lo que esta estructura es.",
            ],
            "ar": [
                "ما يُحدّد هذا المبنى هو كونه مبنى شقق سكنية.",
                "مبنى شقق سكنية هو ما يكون عليه هذا المبنى.",
            ],
            "zh": [
                "这栋建筑的定义特征是它是一栋住宅公寓楼。",
                "住宅公寓楼就是这栋建筑的类型。",
            ],
        },
        "F4": {
            "de": [
                "Dieses Gebäude enthält mehrere Wohneinheiten, in denen Menschen leben.",
                "Es ist ein Gebäude mit mehreren Einheiten, in dem die Bewohner in separaten Wohnungen leben.",
            ],
            "es": [
                "Este edificio contiene múltiples unidades habitacionales para que la gente viva.",
                "Es una estructura de múltiples unidades donde las personas residen en apartamentos separados.",
            ],
            "ar": [
                "يحتوي هذا المبنى على عدة وحدات سكنية ليعيش فيها الناس.",
                "إنه مبنى متعدد الوحدات يسكن فيه الناس في شقق منفصلة.",
            ],
            "zh": [
                "这栋建筑包含多个供人居住的住宅单元。",
                "它是一栋多单元建筑，居民住在各自独立的公寓中。",
            ],
        },
        "F5": {
            "de": [
                "Als Wohnhaus mit Apartments bietet es mehreren Haushalten eine Unterkunft.",
                "Dieses Gebäude ist dafür gebaut, vielen Menschen in separaten Wohneinheiten ein Zuhause zu bieten.",
            ],
            "es": [
                "Como edificio residencial de apartamentos, proporciona vivienda a múltiples hogares.",
                "Esta estructura está construida para servir de hogar a muchas personas en unidades de apartamento separadas.",
            ],
            "ar": [
                "بوصفه مبنى شقق سكنية، يوفّر السكن لعدة أسر.",
                "بُني هذا المبنى ليكون مسكناً للعديد من الأشخاص في وحدات شقق منفصلة.",
            ],
            "zh": [
                "作为住宅公寓楼，它为多个家庭提供住所。",
                "这栋建筑旨在为许多人提供独立公寓单元的住房。",
            ],
        },
    },

    "E4": {
        "F2": {
            "de": [
                "Es ist ein hohes Wohngebäude mit Apartments.",
                "Dieses Wohnhaus ragt viele Stockwerke in die Höhe.",
            ],
            "es": [
                "Es una estructura residencial alta que contiene apartamentos.",
                "Este edificio de apartamentos se eleva muchos pisos.",
            ],
            "ar": [
                "إنه مبنى سكني مرتفع يحتوي على شقق.",
                "يرتفع مبنى الشقق هذا طوابق عديدة.",
            ],
            "zh": [
                "它是一栋包含公寓的高层住宅建筑。",
                "这栋公寓楼高达许多层。",
            ],
        },
        "F3": {
            "de": [
                "Was dieses Gebäude ausmacht, ist, dass es ein hohes Wohnhaus ist.",
                "Ein Hochhaus mit Wohnungen ist das, was dieses Gebäude ist.",
            ],
            "es": [
                "Lo que define este edificio es ser un edificio de apartamentos alto.",
                "Un edificio de apartamentos de gran altura es lo que esta estructura es.",
            ],
            "ar": [
                "ما يُحدّد هذا المبنى هو كونه مبنى شقق شاهقاً.",
                "برج سكني شاهق هو ما يكون عليه هذا المبنى.",
            ],
            "zh": [
                "这栋建筑的定义特征是它是一栋高层公寓楼。",
                "高层公寓楼就是这栋建筑的类型。",
            ],
        },
        "F4": {
            "de": [
                "Dieses hohe Gebäude enthält viele Stockwerke mit Wohnungen.",
                "Es ist ein sehr hohes Gebäude, in dem viele Familien in übereinanderliegenden Wohnungen leben.",
            ],
            "es": [
                "Este edificio alto contiene muchos pisos de apartamentos residenciales.",
                "Es una estructura muy alta donde muchas familias viven en apartamentos apilados.",
            ],
            "ar": [
                "يحتوي هذا المبنى المرتفع على طوابق عديدة من الشقق السكنية.",
                "إنه مبنى شاهق جداً تعيش فيه عائلات كثيرة في شقق متراصّة.",
            ],
            "zh": [
                "这栋高楼包含许多层公寓住宅。",
                "它是一栋非常高的建筑，许多家庭住在层层叠叠的公寓中。",
            ],
        },
        "F5": {
            "de": [
                "Als Hochhaus mit Wohnungen beherbergt es viele Bewohner über viele Stockwerke hinweg.",
                "Dieses mehrstöckige Gebäude enthält Wohnungen auf seinen vielen Etagen.",
            ],
            "es": [
                "Como edificio de apartamentos de gran altura, aloja a muchos residentes en muchos pisos.",
                "Esta estructura de varios pisos contiene apartamentos a lo largo de sus muchos niveles.",
            ],
            "ar": [
                "بوصفه برجاً سكنياً شاهقاً، يأوي العديد من السكان عبر طوابق كثيرة.",
                "يحتوي هذا المبنى متعدد الطوابق على شقق في مستوياته العديدة.",
            ],
            "zh": [
                "作为高层公寓楼，它在许多楼层中容纳了众多居民。",
                "这栋多层建筑在其各个楼层中设有公寓。",
            ],
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN F — Institutional / legal
    # ══════════════════════════════════════════════════════════════════════════

    "F1": {
        "F2": {
            "de": [
                "Es ist eine öffentlich finanzierte Universität mit einem starken Forschungsauftrag.",
                "Die Universität gehört zum öffentlichen Sektor und konzentriert sich auf Forschung.",
            ],
            "es": [
                "Es una universidad financiada con fondos públicos con una fuerte misión investigadora.",
                "La universidad pertenece al sector público y se centra en la investigación.",
            ],
            "ar": [
                "إنها جامعة ممولة من القطاع العام ذات رسالة بحثية قوية.",
                "تنتمي الجامعة إلى القطاع العام وتركّز على البحث.",
            ],
            "zh": [
                "它是一所有着强大科研使命的公立大学。",
                "这所大学属于公共部门，以科研为重点。",
            ],
        },
        "F3": {
            "de": [
                "Was diese Institution ausmacht, ist, dass sie eine öffentliche Forschungsuniversität ist.",
                "Eine öffentliche Forschungsuniversität ist das, was diese Institution ist.",
            ],
            "es": [
                "Lo que define a esta institución es ser una universidad pública de investigación.",
                "Una universidad pública de investigación es lo que esta institución es.",
            ],
            "ar": [
                "ما يُحدّد هذه المؤسسة هو كونها جامعة بحثية حكومية.",
                "جامعة بحثية حكومية هي ما تكون عليه هذه المؤسسة.",
            ],
            "zh": [
                "这所机构的定义特征是它是一所公立研究型大学。",
                "公立研究型大学就是这所机构的类型。",
            ],
        },
        "F4": {
            "de": [
                "Dies ist eine staatlich finanzierte Universität, an der bedeutende Forschung stattfindet.",
                "Es ist eine öffentlich getragene Hochschuleinrichtung mit bedeutenden Forschungsprogrammen.",
            ],
            "es": [
                "Esta es una universidad financiada por el estado donde se realiza investigación significativa.",
                "Es una institución de educación superior de apoyo público con importantes programas de investigación.",
            ],
            "ar": [
                "هذه جامعة ممولة من الدولة تُجرى فيها أبحاث مهمة.",
                "إنها مؤسسة تعليم عالٍ مدعومة حكومياً ذات برامج بحثية كبرى.",
            ],
            "zh": [
                "这是一所由国家资助、开展重要科研工作的大学。",
                "它是一所获得公共支持、拥有重大科研项目的高等教育机构。",
            ],
        },
        "F5": {
            "de": [
                "Als öffentliche Forschungsuniversität wird sie vom Staat finanziert und betreibt Forschung.",
                "Diese Institution wird öffentlich finanziert und ist für ihre starke Forschungstätigkeit bekannt.",
            ],
            "es": [
                "Como universidad pública de investigación, es financiada por el estado y produce investigación.",
                "Esta institución es financiada públicamente y conocida por sus sólidas actividades de investigación.",
            ],
            "ar": [
                "بوصفها جامعة بحثية حكومية، تُموَّل من الدولة وتُنتج الأبحاث.",
                "هذه المؤسسة ممولة حكومياً ومعروفة بأنشطتها البحثية القوية.",
            ],
            "zh": [
                "作为公立研究型大学，它由国家资助并从事科研工作。",
                "这所机构由公共资金资助，以其卓越的科研活动闻名。",
            ],
        },
    },

    "F2": {
        "F2": {
            "de": [
                "Es ist eine vom Staat oder öffentlichen Sektor finanzierte Universität.",
                "Die Universität gehört zum öffentlichen Bildungssystem.",
            ],
            "es": [
                "Es una universidad financiada por el estado o el sector público.",
                "La universidad pertenece al sistema de educación pública.",
            ],
            "ar": [
                "إنها جامعة ممولة من الدولة أو القطاع العام.",
                "تنتمي الجامعة إلى نظام التعليم العام.",
            ],
            "zh": [
                "它是一所由国家或公共部门资助的大学。",
                "这所大学属于公共教育体系。",
            ],
        },
        "F3": {
            "de": [
                "Was diese Institution ausmacht, ist, dass sie eine öffentliche Universität ist.",
                "Eine öffentliche Universität ist das, was diese Institution ist.",
            ],
            "es": [
                "Lo que define a esta institución es ser una universidad pública.",
                "Una universidad pública es lo que esta institución es.",
            ],
            "ar": [
                "ما يُحدّد هذه المؤسسة هو كونها جامعة حكومية.",
                "جامعة حكومية هي ما تكون عليه هذه المؤسسة.",
            ],
            "zh": [
                "这所机构的定义特征是它是一所公立大学。",
                "公立大学就是这所机构的类型。",
            ],
        },
        "F4": {
            "de": [
                "Diese Universität wird von der Regierung statt von privaten Quellen finanziert.",
                "Es ist eine staatlich getragene Hochschuleinrichtung.",
            ],
            "es": [
                "Esta universidad es financiada por el gobierno en lugar de fuentes privadas.",
                "Es una institución de educación superior apoyada por el estado.",
            ],
            "ar": [
                "تُموَّل هذه الجامعة من الحكومة وليس من مصادر خاصة.",
                "إنها مؤسسة تعليم عالٍ مدعومة من الدولة.",
            ],
            "zh": [
                "这所大学由政府而非私人来源资助。",
                "它是一所由国家支持的高等教育机构。",
            ],
        },
        "F5": {
            "de": [
                "Als öffentliche Universität wird sie mit öffentlichen Mitteln finanziert und steht Studierenden weitgehend offen.",
                "Diese Institution erhält staatliche Mittel und fungiert als öffentliche Hochschuleinrichtung.",
            ],
            "es": [
                "Como universidad pública, es financiada con dinero público y abierta ampliamente a los estudiantes.",
                "Esta institución recibe fondos estatales y opera como un proveedor de educación superior pública.",
            ],
            "ar": [
                "بوصفها جامعة حكومية، تُموَّل من المال العام وهي مفتوحة للطلاب على نطاق واسع.",
                "تتلقى هذه المؤسسة تمويلاً حكومياً وتعمل كمزوّد للتعليم العالي العام.",
            ],
            "zh": [
                "作为公立大学，它由公共资金资助，面向广大学生开放。",
                "这所机构接受国家拨款，作为公共高等教育提供者运营。",
            ],
        },
    },

    "F3": {
        "F2": {
            "de": [
                "Es ist ein Arbeitsvertrag mit einem festgelegten Enddatum.",
                "Dieser Vertrag stellt einen Arbeitnehmer für einen begrenzten Zeitraum ein.",
            ],
            "es": [
                "Es un acuerdo laboral con una fecha de finalización definida.",
                "Este contrato emplea a un trabajador por un período limitado.",
            ],
            "ar": [
                "إنها اتفاقية عمل بتاريخ انتهاء محدد.",
                "يوظّف هذا العقد عاملاً لفترة محدودة.",
            ],
            "zh": [
                "它是一份有明确终止日期的劳动协议。",
                "这份合同雇用工人从事有限期的工作。",
            ],
        },
        "F3": {
            "de": [
                "Was dieses Dokument ausmacht, ist, dass es ein befristeter Arbeitsvertrag ist.",
                "Ein befristeter Arbeitsvertrag ist das, was dieses Dokument ist.",
            ],
            "es": [
                "Lo que define este documento es ser un contrato de trabajo de duración determinada.",
                "Un contrato de trabajo de duración determinada es lo que este documento es.",
            ],
            "ar": [
                "ما يُحدّد هذه الوثيقة هو كونها عقد عمل محدد المدة.",
                "عقد عمل محدد المدة هو ما تكون عليه هذه الوثيقة.",
            ],
            "zh": [
                "这份文件的定义特征是它是一份固定期限劳动合同。",
                "固定期限劳动合同就是这份文件的类型。",
            ],
        },
        "F4": {
            "de": [
                "Dieses Dokument formalisiert ein Arbeitsverhältnis, das an einem bestimmten Datum endet.",
                "Es ist ein Vertrag, der jemanden für einen festgelegten Zeitraum beschäftigt, nach dem er ausläuft.",
            ],
            "es": [
                "Este documento formaliza un acuerdo laboral que terminará en una fecha específica.",
                "Es un contrato que emplea a alguien por un período determinado, tras el cual expira.",
            ],
            "ar": [
                "تُضفي هذه الوثيقة الطابع الرسمي على ترتيب عمل ينتهي في تاريخ محدد.",
                "إنه عقد يوظّف شخصاً لفترة محددة ينتهي بعدها.",
            ],
            "zh": [
                "这份文件将一个在特定日期结束的工作安排正式化。",
                "它是一份雇用某人从事定期工作的合同，到期后即终止。",
            ],
        },
        "F5": {
            "de": [
                "Als befristeter Vertrag begründet er ein Arbeitsverhältnis, das an einem festgelegten Datum endet.",
                "Dieser Arbeitsvertrag unterscheidet sich von unbefristeten dadurch, dass er ein festgelegtes Ablaufdatum hat.",
            ],
            "es": [
                "Como contrato de duración determinada, establece un empleo que termina en una fecha fijada.",
                "Este contrato de trabajo se diferencia de los permanentes en que tiene una expiración definida.",
            ],
            "ar": [
                "بوصفه عقداً محدد المدة، يُنشئ علاقة عمل تنتهي في تاريخ معين.",
                "يختلف عقد العمل هذا عن العقود الدائمة في أن له تاريخ انتهاء محدداً.",
            ],
            "zh": [
                "作为固定期限合同，它确立了在设定日期终止的雇佣关系。",
                "这份劳动合同与长期合同不同，因为它有明确的到期日。",
            ],
        },
    },

    "F4": {
        "F2": {
            "de": [
                "Es ist eine Vereinbarung, die ein Arbeitsverhältnis regelt.",
                "Das Dokument formalisiert ein Arbeitsverhältnis.",
            ],
            "es": [
                "Es un acuerdo que rige una relación laboral.",
                "El documento formaliza un acuerdo de trabajo.",
            ],
            "ar": [
                "إنها اتفاقية تنظّم علاقة عمل.",
                "تُضفي هذه الوثيقة الطابع الرسمي على ترتيب وظيفي.",
            ],
            "zh": [
                "它是一份规范劳动关系的协议。",
                "该文件将一项工作安排正式化。",
            ],
        },
        "F3": {
            "de": [
                "Was dieses Dokument ausmacht, ist, dass es ein Arbeitsvertrag ist.",
                "Ein Arbeitsvertrag ist das, was dieses Dokument ist.",
            ],
            "es": [
                "Lo que define este documento es ser un contrato de trabajo.",
                "Un contrato de trabajo es lo que este documento es.",
            ],
            "ar": [
                "ما يُحدّد هذه الوثيقة هو كونها عقد عمل.",
                "عقد عمل هو ما تكون عليه هذه الوثيقة.",
            ],
            "zh": [
                "这份文件的定义特征是它是一份劳动合同。",
                "劳动合同就是这份文件的类型。",
            ],
        },
        "F4": {
            "de": [
                "Dieses Dokument legt die formalen Bedingungen fest, unter denen jemand beschäftigt wird.",
                "Es ist eine rechtliche Vereinbarung, die die Bedingungen eines bezahlten Arbeitsverhältnisses festlegt.",
            ],
            "es": [
                "Este documento establece los términos formales bajo los cuales alguien es empleado.",
                "Es un acuerdo legal que establece las condiciones de una relación laboral remunerada.",
            ],
            "ar": [
                "تحدّد هذه الوثيقة الشروط الرسمية التي يُوظَّف بموجبها شخص ما.",
                "إنها اتفاقية قانونية تُرسي شروط علاقة عمل مدفوعة الأجر.",
            ],
            "zh": [
                "这份文件规定了雇用某人的正式条款。",
                "它是一份确立有偿劳动关系条件的法律协议。",
            ],
        },
        "F5": {
            "de": [
                "Als Arbeitsvertrag formalisiert er die Vereinbarung zwischen Arbeitgeber und Arbeitnehmer.",
                "Dieses Dokument legt die Rechte und Pflichten von Arbeitgeber und Arbeitnehmer fest.",
            ],
            "es": [
                "Como contrato de trabajo, formaliza el acuerdo entre empleador y empleado.",
                "Este documento define los derechos y obligaciones tanto del empleador como del trabajador.",
            ],
            "ar": [
                "بوصفه عقد عمل، يُضفي الطابع الرسمي على الاتفاق بين صاحب العمل والموظف.",
                "تحدّد هذه الوثيقة حقوق وواجبات كلٍّ من صاحب العمل والعامل.",
            ],
            "zh": [
                "作为劳动合同，它将雇主与雇员之间的协议正式化。",
                "这份文件规定了雇主和劳动者双方的权利与义务。",
            ],
        },
    },
}
