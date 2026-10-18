"""Narrative passages used as speech stimuli (AI-generated for this project).

Fourteen passages: one per block in task_events.tsv. Each is long enough
(~65 words, 19 s or more at 140 words per minute) to fill the longest block (19 s);
audio is trimmed to the exact block duration.
"""

_OPENINGS = [
    "The lighthouse keeper climbed the spiral stairs every evening at dusk. "
    "She counted the steps out loud, one hundred and twelve of them, because "
    "her father had done the same thing for thirty years. At the top she wiped "
    "the salt from the glass, struck a match, and waited for the ships.",

    "Our neighbor grows tomatoes in old paint buckets along his fence. Every "
    "August he leaves a basket of them on our porch with a note that says the "
    "same thing every year: too many again, please help. We have never once "
    "seen him eat a tomato, and we have stopped asking why.",

    "The train left the city just after midnight, half empty and very quiet. "
    "A man in a grey coat slept with his hat over his eyes, and a child "
    "pressed her face against the window to watch the lights disappear. "
    "Somewhere near the river the conductor began, softly, to whistle.",

    "When the power went out, the whole street came outside at once. People "
    "who had lived next to each other for years introduced themselves for the "
    "first time. Someone brought a guitar, someone else brought candles, and "
    "by the time the lights came back, nobody wanted to go inside.",

    "My grandmother kept every letter she ever received in a cedar box under "
    "her bed. After she died we read them in the kitchen, one by one, late "
    "into the night. Most were ordinary news about weather and cousins, but "
    "a few were from someone none of us had ever heard of.",

    "The bakery on the corner opens at five in the morning, long before the "
    "rest of the town wakes up. The owner says the first hour is the best part "
    "of his day, when the ovens are warm, the radio is low, and the only "
    "customers are night nurses and delivery drivers on their way home.",

    "On the first day of school, the new teacher asked everyone to write down "
    "one thing they were afraid of. She collected the papers, read them "
    "silently, and then put them in a drawer. In June she gave them back, and "
    "most of us could not remember why we had been so worried.",

    "The museum guard had walked past the same painting ten thousand times "
    "before he noticed the small dog hidden in the corner. After that he could "
    "not stop seeing it. He started telling visitors to look for it, and soon "
    "people were coming to the museum just to find the dog.",

    "Every winter the lake froze so thick that trucks could drive across it. "
    "Fishermen built little wooden huts on the ice and stayed out there for "
    "days, playing cards and listening to the ice groan beneath them. In "
    "spring the huts were pulled back to shore before the thaw.",

    "The old radio in the garage only picked up one station, which played "
    "nothing but weather reports and polka music. My uncle refused to replace "
    "it. He said a radio that knew exactly what it wanted was rare, and that "
    "the rest of us could learn something from it.",

    "At the edge of the desert there is a gas station with a single pump and "
    "a sign that promises the best pie in three states. Nobody has ever "
    "checked whether that is true. Truck drivers stop anyway, order a slice, "
    "and argue about it for the next hundred miles.",

    "The library stayed open late on Thursdays, and that was when the chess "
    "players came. They set up their boards on the long tables near the "
    "windows and played in complete silence for hours. The librarian said they "
    "were the quietest and most loyal patrons she had ever known.",

    "We found the kitten in a cardboard box behind the hardware store, wet and "
    "furious. It hissed at everyone for a week, slept in a shoe for a month, "
    "and then decided, for reasons it never explained, that it owned the "
    "house, the yard, and most of the people in it.",

    "The mountain road was closed for the winter, so we walked the last mile "
    "to the cabin through deep snow. By the time we arrived our boots were "
    "soaked and the sun had set. We lit the stove, made soup from whatever we "
    "could find, and fell asleep before we finished eating.",
]

# Closing sentences, appended so every passage outlasts the longest (19 s) block.
_CLOSINGS = [
    "On clear nights the beam reached so far that sailors said they could read by it from the harbor.",
    "My mother says he is the kindest man on the street, and also the most mysterious one.",
    "By the time the sun came up, the train had crossed two rivers and nobody on board was asleep.",
    "The next summer they held the same party on purpose, and they have held it every summer since.",
    "We still argue about whether we should try to find that person, or leave the letters exactly as they were.",
    "He knows every regular by name, and he always saves the last cinnamon roll for the mail carrier.",
    "She kept the drawer locked all year, and to this day no one knows what she wrote on her own paper.",
    "The museum eventually printed the dog on its postcards, and the guard was given his own small brass plaque.",
    "Every year someone waited a little too long, and the whole town would gather on the shore to watch the rescue.",
    "When he finally sold the house, the radio was the only thing he insisted on taking with him.",
    "The owner just smiles, refills the coffee, and says that the argument is the real secret ingredient.",
    "When one of them passed away, the others left his chair empty at the table for an entire year.",
    "Ten years later it is still there, asleep on the warmest windowsill, ruling over all of us.",
    "In the morning the snow was so bright that we had to squint, and we decided to stay another week.",
]

PASSAGES = [f"{o} {c}" for o, c in zip(_OPENINGS, _CLOSINGS, strict=True)]
