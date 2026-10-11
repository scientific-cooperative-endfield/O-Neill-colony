# Отправка проекта на GitHub

Создайте пустой репозиторий GitHub без автоматического README, лицензии и `.gitignore`.
Затем выполните команды из корня проекта (замените OWNER и REPOSITORY):

```bash
git init -b main
git add .
git status --short
git diff --cached --stat
git commit -m "Initial import of O'Neill Colony"
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

Если Git требует имя и email, задайте их для этого репозитория через
`git config user.name` и `git config user.email`; можно использовать свой GitHub noreply email.

Если репозиторий уже инициализирован, пропустите `git init`.
Если `origin` уже задан, проверьте `git remote -v` перед его изменением.

В репозиторий входят `Cargo.lock`, модели `assets/models/*.omdl`,
`assets/fracture.bin`, сценарии, исходники, скрипты Blender и иллюстрации.
Сборки, Python-кеши, HDR-кеш неба, результаты запусков и локальные служебные
каталоги исключены через `.gitignore`.

Перед первым push выполните `cargo test --locked --workspace` и выберите
видимость репозитория. Лицензия остаётся на усмотрение владельца проекта.

Если среда запрещает запись в `.git`, распакуйте подготовленный архив
`oneill-github.tar.gz` в обычный доступный для записи каталог и выполните команды там.
Архив содержит каталог `oneill/` без локальных сборок и служебных файлов.
