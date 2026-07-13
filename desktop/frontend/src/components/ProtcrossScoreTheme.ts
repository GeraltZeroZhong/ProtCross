import { Bond, StructureElement, Unit } from "molstar/lib/mol-model/structure";
import type { ThemeDataContext } from "molstar/lib/mol-theme/theme";
import type { ColorTheme } from "molstar/lib/mol-theme/color";
import { Color, ColorScale } from "molstar/lib/mol-util/color";

const DEFAULT_COLOR = Color(0x66727c);

function scoreFor(unit: Unit, element: number): number {
  if (!Unit.isAtomic(unit)) {
    return 0;
  }
  return unit.model.atomicConformation.B_iso_or_equiv.value(element);
}

export function ProtcrossScoreColorTheme(_ctx: ThemeDataContext, props: Record<string, never>): ColorTheme<{}> {
  const scale = ColorScale.create({
    domain: [0, 1],
    reverse: false,
    minLabel: "0.00",
    maxLabel: "1.00",
    listOrName: [
      [Color(0x173b57), 0],
      [Color(0x2d708e), 0.3],
      [Color(0x5cb6a5), 0.55],
      [Color(0xf2c14e), 0.78],
      [Color(0xe4572e), 1]
    ]
  });

  function color(location: unknown) {
    if (StructureElement.Location.is(location)) {
      return scale.color(scoreFor(location.unit, location.element));
    }
    if (Bond.isLocation(location)) {
      return scale.color(scoreFor(location.aUnit, location.aUnit.elements[location.aIndex]));
    }
    return DEFAULT_COLOR;
  }

  return {
    factory: ProtcrossScoreColorTheme,
    granularity: "group",
    preferSmoothing: true,
    color,
    props,
    description: "ProtCross binding-site model score stored in the annotated structure B-factor field.",
    legend: scale.legend
  };
}

export const ProtcrossScoreColorThemeProvider: ColorTheme.Provider<{}, "protcross-score"> = {
  name: "protcross-score",
  label: "ProtCross score",
  category: "Miscellaneous",
  factory: ProtcrossScoreColorTheme,
  getParams: () => ({}),
  defaultValues: {},
  isApplicable: (ctx) => Boolean(ctx.structure?.models.some((model) => model.atomicConformation.B_iso_or_equiv.isDefined))
};
