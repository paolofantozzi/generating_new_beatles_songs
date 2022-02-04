# Generating new Beatles' songs
A jupyter notebook to fine tune a GPT2 pretrained model in order to generate new Beatles' song lyrics. They are included some examples of lyrics generated in the folder [examples](examples/).

In the notebook, we assume to have a file called `Lyrics_TheBeatles_cleaned.json`, containing all the lyrics written by Beatles in form of a list of dictionaries with keys `title` and `lyrics`. The lyrics are used to perform the fine tuning of the model. It also assumes to have a CUDA device available.

The lyrics has been scraped from [Genius.com](https://genius.com/) by using their APIs. They are not included in the repository to avoid license issues.

The notebook provides a function `generate` that takes as input:
 - title of the song you want to generate
 - temperature parameter
 - top_k parameter

Examples
========

Can't Help Falling in Love __(temperature=0.8, top_k=0)__
----------------------------------------------------------

[Intro]
Can't help falling in love
I can't help it

[Verse 1]
I'm falling in love with you
I can't help it
But I'm always wondering why
It's so hard for a girl like you to hold me
Can't help falling in love

[Chorus]
I'm falling in love for you (I'm falling in love)
I'm falling in love for you (I'm falling in love)
You're falling in love for me (Oh my darling)

[Verse 2]
I'm falling in love with you
Why, do you think I need you?
I can't help it
But I'm always wondering why
It's so hard for a girl like you to hold me
Can't help falling in love

[Bridge]
I'm falling in love for you (I'm falling in love)
You're falling in love for me (Oh my darling)
You're falling in love for me (Oh my darling)

I don't want to miss a thing __(temperature=0.8, top_k=40)__
------------------------------------------------------------

[Verse 1]
I don't want to miss a thing
It's only a whim, but a day or two goes by
I don't care too much for money
Money can't buy me love

[Verse 2]
I've got no time for trivialities
I've lost all hope
When every moment feels so...
It's only a whim, but a day or two goes by
I don't care too much for money
Money can't buy me love

[Chorus]
I don't want to miss a thing
It's only a whim, but a day or two goes by
I don't care too much for money
Money can't buy me love

[Verse 3]
You know I hate to see the sun
Sinking to the west
'Cause I know my whole life's
Gonna be one solid mess

[Chorus]
I don't want to miss a thing
It's only a whim, but a day or two goes by
I don't care too much for money
Money can't buy me love

[Verse 4]
You know I hate to see the sun
Falling in love again
Sinking to the west
'Cause I know my whole life's
Gonna be one solid mess

[Chorus]
I don't want to miss a thing
It's only a whim, but a day or two goes by
I don't care too much for money
Money can't buy me love

[Outro]
Ooooh oh
Yeah! Ow! Ow!

[Verse 5]
You know I hate to see the sun
Sinking to the west
'Cause I know my whole life's
Gonna be one solid mess

[Outro]
Aw! Ow! Ow!

[Chorus]
I don't want to miss a thing
It's only a whim, but a day or two goes by
I don't care too much for money
Money can't buy me love

[Verse 6]
You know I hate to see the sun
Sinking to the west
'Cause I know my whole life's
Gonna be one solid mess

[Outro]
Ooooh oh
Yeah! Ow!

[Verse 7]
You know I hate to see the sun
Thrusting it between my legs
Keeping my head down, I cry out loud
Hey! Yeah! Ow!

[Outro]
Aw! Ow!

Isn't she lovely __(temperature=0.8, top_k=40)__
------------------------------------------------

[Intro]

[Verse 1]
Isn't she good to me?
I think of her, and so do I
Love her, and so should you

[Verse 2]
Isn't she sweet and lovely
She tastes as Mongolian lamb
Tasting all my dreams

[Bridge]
Isn't she the kind of girl I love?
People tell me she's only
Fooling, I know she isn't

[Verse 3]
Isn't she good to me?
I think of her, and so do I
She tastes as Mongolian lamb
Tasting all my dreams

[Bridge]
Isn't she the kind of girl I love?
People tell me she's only
Fooling, I know she isn't

[Verse 4]
Isn't she good to me?
I think of her, and so do I
She tastes as Mongolian lamb
Tasting all my dreams

[Bridge]
Isn't she the kind of girl I love?
People tell me she's only
Fooling, I know she isn't

[Verse 5]
Isn't she good to me?
I think of her, and so do I
She tastes as Mongolian lamb
Tasting all my dreams

[Bridge]
Isn't she the kind of girl I love?
People tell me she's only
Fooling, I know she isn't

[Verse 6]
Isn't she good to me?
I think of her, and so do I
She tastes as Mongolian lamb
Tasting all my dreams

[Bridge]
Isn't she the kind of girl I love?
People tell me she's only
Fooling, I know she isn't

[Verse 7]
Isn't she good to me?
I think of her, and so do I
She tastes as Mongolian lamb
Tasting all my dreams

[Bridge]
Isn't she the kind of girl I love?
People tell me she's only
Fooling, I know she isn't

[Verse 8]
Isn't she good to me?
I think of her, and so do I
She tastes as Mongolian lamb
Tasting all my dreams

[Bridge]
Isn't she the kind of girl I love?
People tell me she's only
Fooling, I know she isn't

[Verse 9]
Isn't she good to me?
I think of her, and so do I
She tastes as Mongolian lamb
Tasting all my dreams

[Bridge]
Isn't she the kind of girl I love?
People tell me she's only
Fooling, I know she isn't

[Verse 10]
Isn't she good to me?
I think of her, and so do I
She tastes as Mongolian lamb
Tasting all my dreams

[Bridge]
Isn't she the kind of girl I love?
People tell me she's only
Fooling, I know she isn't

[Verse 11]
Isn't she good to me?
I think of her, and so do I
She tastes as Mongolian lamb
Tasting all my dreams

[Bridge]
Isn't she the kind of girl I love?
People tell me she's only
Fooling, I know she isn't

[Verse 12]
Isn't she good to me?
I think of her, and so do I
She tastes as Mongolian lamb
Tasting all my dreams

[Bridge]
Isn't she the kind of girl I love?
People tell me she's only
Fooling, I know she isn't

[Verse 13]
Isn't she good to me?
I think of her, and so do I
She tastes as Mongolian lamb
Tasting all my dreams

[Bridge]
Isn't she the kind of girl I love?
People tell me she's only
Fooling, I know she isn't
