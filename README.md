# bonjour

> [!CAUTION]
>
> Work in progress, first usable version hasn't landed yet!

Experimental crate to extract first names with a confidence level.

Expected output may be something like this:

```json
{
  "input": "Quentin Richert",
  "first_name": "Quentin",
  "confidence": 0.95,
  "gender": "male",
  "country": "FR"
}
```

The idea is that is also "detects", or at lease significantly reduces
confidences in company names, for instance:

```json
// The company marker 'SAS' significantly reduces confidence.
{
  "input": "Quentin Richert SAS",
  "first_name": "Quentin",
  "confidence": 0.1,
  "gender": "male",
  "country": "FR"
}
```

Up to no detection at all:

```json
{
  "input": "Les Motards d'Alsace",
  "first_name": null,
  "confidence": 0.0,
  "gender": null,
  "country": null
}
```

## Country and gender hints

Gender is not a property of a name alone — `Simone` is female in France,
male in Italy. Pass the user's country and/or gender as hints and they
resolve each other: a country pins the gender, a gender pins the
country.

```console
$ bonjour --country=IT Simone Veil
{
  "input": "Simone Veil",
  "first_name": "Simone",
  "confidence": 0.65,
  "gender": "male",
  "country": "IT"
}

$ bonjour --country=FR Simone Veil
{
  "input": "Simone Veil",
  "first_name": "Simone",
  "confidence": 0.7,
  "gender": "female",
  "country": "FR"
}
```

With no hint and a name whose gender differs by country, `gender` is
left `null` rather than guessed:

```json
{
  "input": "Simone",
  "first_name": "Simone",
  "confidence": 0.7,
  "gender": null,
  "country": "FR"
}
```

## License

The source code is available under the [0BSD license](LICENSE).

Datasets distributed with or used to build this project are compiled
from publicly available information. They are not covered by the 0BSD
license; their contents remain subject to any applicable rights and
source terms.
