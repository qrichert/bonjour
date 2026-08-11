# bonjour

> [!CAUTION]
>
> Work in progress, first usable version hasn't landed yet!

Experimental crate to extract first names from display names.

```console
$ bonjour "Quentin Richert"
Bonjour Quentin !
```

## Use Case

In social apps, it's often nicer UX to store display names instead of
separate first and last names.

The reason being, if you're a business or an association, filling-in
"first name" and "last name" is awkward, and if you have OCD like me,
this would be a major annoyance:

```
> First Name: Motorcycle
> Last Name: Club
```

Especially if the UI later greeted you like:

```
Hello Motorcycle!
```

### The Theoretical "Better" Way

Instead you could store a display name, which solves the greeting
problem for entities:

```
> Display Name: Motorcycle Club
> Hello Motorcycle Club!
```

But creates a new one for people:

```
> Display Name: Quentin Richert
```

You could greet people with their full name, but it's a bit unnatural
and less warm.

Instead, you'd want to greet them by their first name if you could
_identify their first name with high confidence_; which is exactly the
point of this project.

```console
$ bonjour --json "Quentin Richert"
{
  "input": "Quentin Richert",
  "first_name": "Quentin",
  "confidence": 0.95,
  "gender": "male",
  "country": "FR"
}
```

There's a very high chance "Quentin" is the first name here, so it's
overwhelmingly fine to write:

```
Hello Quentin!
```

## Why it's not that simple

- `display_name.split_whitespace()[0]` Well, some countries don't follow
  the _first-name-last-name_ convention, and even in countries that _do_
  follow it, some people may not. Moreover, it risks extracting "The"
  from "The Motorcycle Club".
- **A dictionary of first names.** First names can also be last names,
  for instance in France, is both a popular first name and a popular
  last name. So in "Jean Martin", which one is the first name?
- Another caveat is that in certain languages, the greeting agrees in
  gender and in number, which means you also have to know the gender of
  the name for a proper greeting. However, the same name can have a
  different gender based on the country the person is in. In France,
  "Simone" is unequivocally a woman's name, but, in Italy, "Simone" is a
  man's name (Simone (FR) = Simona (IT) and Simone (IT) = Simon (FR)).

... and many other things that can only be statistically answered.
That's why `bonjour` returns confidence levels, and it's up to the user
to determine if the confidence is high enough for the use case.

To help guide the detection, `bonjour` accepts hints. because if you
know the country, or the gender, or both, it massively increases the
confidence in its detections.

## Usage

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
