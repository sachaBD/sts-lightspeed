#pragma once
#include "combat/CardPlay.h"

namespace sts {
enum class ResolutionPhase {
    InitialToolbox, InitialGamble,
    CardPrepare, CardEffects, CardAfterChoice, CardTriggers, CardCleanup, CardDone,
    EndRelics, EndPowers, EndDiscard, Monsters, RoundEnd, StartPowers, StartDraw, StartDiscards,
    Done
};

struct ResolutionFrame {
    ResolutionPhase phase = ResolutionPhase::Done;
    CardPlay card;
    int drawsAfterChoice = 0;
    int remainingDiscards = 0;

    bool isCard() const {
        return phase >= ResolutionPhase::CardPrepare && phase <= ResolutionPhase::CardDone;
    }
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.phase); visit(s.card); visit(s.drawsAfterChoice); visit(s.remainingDiscards);
    }
};
}
